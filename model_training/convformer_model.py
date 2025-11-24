import torch
import torch.nn as nn
from transformers import Wav2Vec2ConformerConfig, Wav2Vec2ConformerModel

class ConvFormerDecoder(nn.Module):
    """
    Defines the ConvFormer (Conformer) Decoder.
    """
    def __init__(self, 
                 neural_dim, 
                 n_units, 
                 n_days, 
                 n_classes, 
                 rnn_dropout=0.1, 
                 input_dropout=0.0, 
                 n_layers=6, 
                 patch_size=0, 
                 patch_stride=0,
                 n_heads=8,
                 **kwargs): 
        super().__init__()
        
        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_days = n_days
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.input_dropout = input_dropout

        # --- 1. Day-Specific Input Layers ---
        self.day_layer_activation = nn.Softsign() 

        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )

        self.day_layer_dropout = nn.Dropout(input_dropout)
        
        # --- 2. Input Dimension Calculation ---
        self.input_size = self.neural_dim
        if self.patch_size > 0:
            self.input_size *= self.patch_size

        # --- 3. Input Projection ---
        self.input_projection = nn.Linear(self.input_size, n_units)

        # --- 4. Conformer Configuration ---
        config = Wav2Vec2ConformerConfig(
            hidden_size=n_units,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            hidden_dropout=rnn_dropout,
            attention_dropout=rnn_dropout,
            feat_proj_dropout=rnn_dropout,
            intermediate_size=n_units * 4, 
            conformer_conv_dropout=rnn_dropout,
            hidden_act="swish",
            position_embeddings_type="relative",
        )
        
        self.conformer = Wav2Vec2ConformerModel(config)
        
        # --- 5. Output Head ---
        self.out = nn.Linear(n_units, n_classes)

        nn.init.xavier_uniform_(self.out.weight)
        nn.init.xavier_uniform_(self.input_projection.weight)

    def forward(self, x, day_idx, states=None, return_state=False):
        # 1. Day Layers
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        # 2. Patching
        if self.patch_size > 0:
            x = x.unsqueeze(1)
            x = x.permute(0, 3, 1, 2)
            x_unfold = x.unfold(3, self.patch_size, self.patch_stride) 
            x_unfold = x_unfold.squeeze(2)
            x_unfold = x_unfold.permute(0, 2, 3, 1)
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1) 

        # 3. Backbone
        x = self.input_projection(x)
        
        # FIX: Call .encoder() directly to pass hidden_states instead of inputs_embeds
        outputs = self.conformer.encoder(x)
        x = outputs.last_hidden_state

        # 4. Output
        logits = self.out(x)
        
        if return_state:
            return logits, None
            
        return logits


def build_model():
    # This helper mimics your previous structure
    # You can adjust arguments here for testing
    model = ConvFormerDecoder(
        neural_dim=256,
        n_units=512,
        n_days=1,       # Placeholder
        n_classes=40,   # Placeholder
        n_layers=6
    )
    return model, None