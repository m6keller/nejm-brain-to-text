import torch
from torch import nn
import math


class DayInputLayer(nn.Module):
    """
    Shared day-specific input layer (same idea as GRUDecoder).
    """
    def __init__(self, neural_dim, n_days, input_dropout=0.0):
        super().__init__()
        self.neural_dim = neural_dim
        self.n_days = n_days
        self.day_layer_activation = nn.Softsign()
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )
        self.input_dropout = input_dropout
        self.day_layer_dropout = nn.Dropout(input_dropout)

    def forward(self, x, day_idx):
        # x: [batch, time, neural_dim]
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)
        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)
        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)
        return x


def apply_patching(x, patch_size, patch_stride):
    """
    Reuse the same patching logic as GRUDecoder.
    x: [batch, time, feat]
    returns: [batch, num_patches, patch_size * feat]
    """
    if patch_size <= 0:
        return x

    x = x.unsqueeze(1)  # [b, 1, t, f]
    x = x.permute(0, 3, 1, 2)  # [b, f, 1, t]
    x_unfold = x.unfold(3, patch_size, patch_stride)  # [b, f, 1, num_patches, patch_size]
    x_unfold = x_unfold.squeeze(2)  # [b, f, num_patches, patch_size]
    x_unfold = x_unfold.permute(0, 2, 3, 1)  # [b, num_patches, patch_size, f]
    x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1)
    return x


# ----------------------
# 1) Temporal CNN (TCN)
# ----------------------

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity="relu")
        if self.downsample is not None:
            nn.init.kaiming_uniform_(self.downsample.weight, nonlinearity="linear")

    def forward(self, x):
        # x: [b, c, t]
        out = self.conv1(x)
        out = out[:, :, :-self.conv1.padding[0]]  # remove extra padding
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]]
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return torch.relu(out + res)


class TCNDecoder(nn.Module):
    """
    Temporal Convolutional Network decoder with day-specific input layer.
    """
    def __init__(self,
                 neural_dim,
                 n_units,
                 n_days,
                 n_classes,
                 rnn_dropout=0.0,      # use as TCN dropout
                 input_dropout=0.0,
                 n_layers=4,
                 patch_size=0,
                 patch_stride=1,
                 kernel_size=3):
        super().__init__()

        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_days = n_days
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        self.day_input = DayInputLayer(neural_dim, n_days, input_dropout)

        # input channels after patching
        self.input_size = neural_dim if patch_size <= 0 else neural_dim * patch_size

        layers = []
        in_channels = self.input_size
        for i in range(n_layers):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=n_units,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=rnn_dropout,
                )
            )
            in_channels = n_units
        self.tcn = nn.Sequential(*layers)

        self.out = nn.Linear(n_units, n_classes)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x, day_idx, states=None, return_state=False):
        # x: [b, t, neural_dim]
        x = self.day_input(x, day_idx)
        x = apply_patching(x, self.patch_size, self.patch_stride)  # [b, t', input_size]

        x = x.transpose(1, 2)  # [b, c, t']
        y = self.tcn(x)        # [b, n_units, t']
        y = y.transpose(1, 2)  # [b, t', n_units]

        logits = self.out(y)   # [b, t', n_classes]
        if return_state:
            return logits, None
        return logits


# --------------------------
# 2) Transformer-based model
# --------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [b, t, d_model]
        t = x.size(1)
        return x + self.pe[:, :t, :]


class TransformerDecoder(nn.Module):
    """
    Small Transformer encoder for neural-to-phoneme decoding.
    """
    def __init__(self,
                 neural_dim,
                 n_units,
                 n_days,
                 n_classes,
                 rnn_dropout=0.1,      # use as Transformer dropout
                 input_dropout=0.0,
                 n_layers=4,
                 patch_size=0,
                 patch_stride=1,
                 n_heads=4,
                 dim_feedforward=1024):
        super().__init__()

        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_days = n_days
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        self.day_input = DayInputLayer(neural_dim, n_days, input_dropout)

        self.input_size = neural_dim if patch_size <= 0 else neural_dim * patch_size

        self.input_proj = nn.Linear(self.input_size, n_units)
        self.pos_encoding = PositionalEncoding(n_units)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_units,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=rnn_dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.out = nn.Linear(n_units, n_classes)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x, day_idx, states=None, return_state=False):
        # x: [b, t, neural_dim]
        x = self.day_input(x, day_idx)
        x = apply_patching(x, self.patch_size, self.patch_stride)  # [b, t', input_size]

        x = self.input_proj(x)         # [b, t', n_units]
        x = self.pos_encoding(x)       # [b, t', n_units]

        # No attention mask (full context)
        y = self.encoder(x)            # [b, t', n_units]

        logits = self.out(y)           # [b, t', n_classes]
        if return_state:
            return logits, None
        return logits
