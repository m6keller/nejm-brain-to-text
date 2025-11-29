import math
import torch
from torch import nn


class DayInputLayer(nn.Module):
    """
    Day-specific input transform with Softsign + dropout.
    Matches the style used in GRUDecoder / previous variants.
    """
    def __init__(self, neural_dim, n_days, input_dropout=0.0):
        super().__init__()
        self.neural_dim = neural_dim
        self.n_days = n_days
        self.act = nn.Softsign()
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(neural_dim)) for _ in range(n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, neural_dim)) for _ in range(n_days)]
        )
        self.dropout_p = input_dropout
        self.dropout = nn.Dropout(input_dropout)

    def forward(self, x, day_idx):
        # x: [B, T, D]
        # day_idx: [B] (long)
        W = torch.stack([self.day_weights[i] for i in day_idx], dim=0)  # [B, D, D]
        b = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)  # [B, 1, D]
        x = torch.einsum("btd,bdk->btk", x, W) + b
        x = self.act(x)
        if self.dropout_p > 0.0:
            x = self.dropout(x)
        return x


def apply_patching(x, patch_size, patch_stride):
    """
    Same semantics as GRUDecoder patching: concatenate patch_size timesteps
    with stride patch_stride along time. If patch_size <= 0, return x.
    x: [B, T, D] -> [B, T', patch_size*D]
    """
    if patch_size <= 0:
        return x

    B, T, D = x.shape
    # [B, D, T]
    x = x.permute(0, 2, 1).contiguous()
    # Unfold over time (dim=2)
    x = x.unfold(dimension=2, size=patch_size, step=patch_stride)  # [B, D, T', patch_size]
    # [B, T', D, patch_size]
    x = x.permute(0, 2, 1, 3).contiguous()
    # [B, T', D*patch_size]
    x = x.view(B, x.size(1), -1)
    return x


# -------------------------------------------------------
# 1) Temporal Convolutional Recurrent Network (TCRN-like)
#    CNN encoder -> small BiGRU stack -> linear head.
# -------------------------------------------------------

class TCRNEncoder(nn.Module):
    """
    Temporal convolutional encoder: 1D conv stack with increasing dilation.
    """
    def __init__(self, in_channels, hidden_channels, n_layers, kernel_size=5, dropout=0.1):
        super().__init__()
        layers = []
        c_in = in_channels
        for i in range(n_layers):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            conv = nn.Conv1d(
                c_in, hidden_channels, kernel_size,
                padding=padding, dilation=dilation
            )
            layers.append(
                nn.Sequential(
                    conv,
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
            )
            c_in = hidden_channels
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # x: [B, C, T]
        for layer in self.layers:
            y = layer(x)
            # remove extra padding from causal conv
            pad = layer[0].padding[0]
            if pad > 0:
                y = y[:, :, :-pad]
            x = y
        return x  # [B, C, T]


class TCRNDecoder(nn.Module):
    """
    Temporal Conv + BiGRU hybrid for neural-to-phoneme decoding.
    - DayInputLayer + (optional) patching
    - Temporal CNN encoder
    - BiGRU stack
    - Linear classifier over time
    """
    def __init__(
        self,
        neural_dim,
        n_units,
        n_days,
        n_classes,
        rnn_dropout=0.1,      # used for GRU + encoder dropout
        input_dropout=0.0,
        n_layers=4,           # total depth: we split between CNN and GRU
        patch_size=0,
        patch_stride=1,
        cnn_layers=2,         # how many conv layers in encoder
        gru_layers=2          # how many BiGRU layers
    ):
        super().__init__()
        assert cnn_layers + gru_layers == n_layers, \
            "For simplicity, require cnn_layers + gru_layers == model.n_layers"

        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_days = n_days
        self.n_classes = n_classes
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.cnn_layers = cnn_layers
        self.gru_layers = gru_layers

        self.day_input = DayInputLayer(neural_dim, n_days, input_dropout)

        self.input_size = neural_dim if patch_size <= 0 else neural_dim * patch_size

        # CNN encoder on channel dimension
        self.encoder = TCRNEncoder(
            in_channels=self.input_size,
            hidden_channels=n_units,
            n_layers=cnn_layers,
            kernel_size=5,
            dropout=rnn_dropout
        )

        # BiGRU on encoder outputs
        self.bigru = nn.GRU(
            input_size=n_units,
            hidden_size=n_units,
            num_layers=gru_layers,
            batch_first=True,
            dropout=rnn_dropout if gru_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.out = nn.Linear(2 * n_units, n_classes)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x, day_idx, states=None, return_state=False):
        # x: [B, T, neural_dim]
        x = self.day_input(x, day_idx)
        x = apply_patching(x, self.patch_size, self.patch_stride)  # [B, T', D_in]

        # CNN expects [B, C, T']; treat features as channels
        x = x.transpose(1, 2)  # [B, D_in, T']
        x = self.encoder(x)    # [B, n_units, T']
        x = x.transpose(1, 2)  # [B, T', n_units]

        if states is not None:
            output, h = self.bigru(x, states)
        else:
            output, h = self.bigru(x)

        logits = self.out(output)  # [B, T', n_classes]
        if return_state:
            return logits, h
        return logits


# -----------------------------------------------------------
# 2) EEG-style Conv-Transformer (local CNN + Transformer)
#    Inspired by EEGConvTransformer/TCFormer style hybrids.
# -----------------------------------------------------------

class ConvFeatureExtractor(nn.Module):
    """
    Lightweight temporal CNN to produce local features before Transformer.
    """
    def __init__(self, in_dim, hidden_dim, kernel_size=7, n_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        c_in = in_dim
        for i in range(n_layers):
            conv = nn.Conv1d(
                in_channels=c_in,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            layers.append(
                nn.Sequential(
                    conv,
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
            )
            c_in = hidden_dim
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # x: [B, C, T]
        for layer in self.layers:
            x = layer(x)
        return x  # [B, hidden_dim, T]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
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
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:, :T, :]


class EEGConvTransformerDecoder(nn.Module):
    """
    CNN + Transformer hybrid for neural decoding.
    - DayInputLayer + patching
    - Temporal CNN feature extractor
    - TransformerEncoder
    - Linear classifier
    """
    def __init__(
        self,
        neural_dim,
        n_units,
        n_days,
        n_classes,
        rnn_dropout=0.1,      # used as Transformer dropout
        input_dropout=0.0,
        n_layers=4,
        patch_size=0,
        patch_stride=1,
        n_heads=4,
        dim_feedforward=1024,
        cnn_layers=2
    ):
        super().__init__()

        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_days = n_days
        self.n_classes = n_classes
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        self.day_input = DayInputLayer(neural_dim, n_days, input_dropout)

        self.input_size = neural_dim if patch_size <= 0 else neural_dim * patch_size

        # Local temporal CNN (on channels) before Transformer
        self.cnn = ConvFeatureExtractor(
            in_dim=self.input_size,
            hidden_dim=n_units,
            kernel_size=7,
            n_layers=cnn_layers,
            dropout=rnn_dropout,
        )

        self.pos_encoder = PositionalEncoding(n_units)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=n_units,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=rnn_dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.out = nn.Linear(n_units, n_classes)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x, day_idx, states=None, return_state=False):
        # x: [B, T, neural_dim]
        x = self.day_input(x, day_idx)
        x = apply_patching(x, self.patch_size, self.patch_stride)  # [B, T', D_in]

        # CNN in [B, C, T'] space
        x = x.transpose(1, 2)                # [B, D_in, T']
        x = self.cnn(x)                      # [B, n_units, T']
        x = x.transpose(1, 2)                # [B, T', n_units]

        x = self.pos_encoder(x)              # [B, T', n_units]
        y = self.encoder(x)                  # [B, T', n_units]

        logits = self.out(y)                 # [B, T', n_classes]
        if return_state:
            return logits, None
        return logits