import torch
from torch import nn

from transformers import MambaConfig, MambaModel

class MambaDecoder(nn.Module):
    """
    Defines the Mamba (State Space Model) Decoder.
    
    This replaces the Conformer backbone with Mamba.
    Mamba is an SSM that scales linearly with sequence length (fast inference),
    unlike Transformers which scale quadratically.
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
                 **kwargs): # Mamba doesn't use 'n_heads'
        super().__init__()
        
        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_days = n_days
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.input_dropout = input_dropout

        # --- 1. Day-Specific Input Layers (Identical logic) ---
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

        # --- 4. Mamba Configuration ---
        # Mamba Config mapping
        config = MambaConfig(
            hidden_size=n_units, 
            num_hidden_layers=n_layers, 
            use_bias=True,
            use_cache=False
            # Mamba doesn't have standard "dropout" args in the config the same way BERT does,
            # but we can manually apply dropout after the backbone if needed.
        )
        
        # The Body (Mamba Backbone)
        self.mamba = MambaModel(config)
        
        # --- 5. Output Head ---
        self.out = nn.Linear(n_units, n_classes)

        # Weight Initializations
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.xavier_uniform_(self.input_projection.weight)

    def forward(self, x, day_idx, states=None, return_state=False):
        """
        x: (batch_size, time_series_length, neural_dim)
        day_idx: (batch_size)
        """

        # --- 1. Apply Day-Specific Layers ---
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        # --- 2. Patching / Striding ---
        if self.patch_size > 0:
            x = x.unsqueeze(1)
            x = x.permute(0, 3, 1, 2)
            x_unfold = x.unfold(3, self.patch_size, self.patch_stride) 
            x_unfold = x_unfold.squeeze(2)
            x_unfold = x_unfold.permute(0, 2, 3, 1)
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1) 

        # --- 3. Mamba Backbone ---
        # Project to Mamba's hidden size
        x = self.input_projection(x)
        
        # Mamba forward pass
        # Note: Mamba is causal. It processes left-to-right.
        outputs = self.mamba(inputs_embeds=x)
        x = outputs.last_hidden_state

        # --- 4. Output Head ---
        logits = self.out(x)
        
        if return_state:
            # Mamba maintains internal state (cache_params), but for simple training we usually ignore it
            return logits, None
            
        return logits
