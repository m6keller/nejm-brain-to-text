import torch
import torch.nn as nn
from transformers import Wav2Vec2ConformerModel, Wav2Vec2ConformerConfig

class RelativeConvFormer(nn.Module):
    def __init__(self, 
                 input_dim,          # Neural channels (e.g., 256)
                 num_classes,        # Phonemes (e.g., 41)
                 n_days,             # Total number of recording days
                 pretrained_model="facebook/wav2vec2-conformer-rel-pos-large",
                 patch_size=1,
                 patch_stride=1,     #
                 input_dropout=0.1,  # Dropout for the day layer
                 n_layers=12,
                 n_attention_heads=8,
                 intermediate_size=1024,
                 hidden_size=512,
                 freeze_backbone=False):
        super().__init__()
        
        self.patch_size = patch_size
        self.patch_stride = patch_stride # <--- Store stride
        self.input_dropout = input_dropout
        self.n_days = n_days
        self.n_layers = n_layers
        
        # --- 1. Day-Specific Input Layer ---
        # We create a unique weight matrix (Identity init) and bias (Zero init) for each day.
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(input_dim)) for _ in range(n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, input_dim)) for _ in range(n_days)]
        )
        
        self.day_layer_activation = nn.Softsign()
        self.day_layer_dropout = nn.Dropout(input_dropout)
        self.intermediate_size = intermediate_size
        self.n_attention_heads = n_attention_heads
        self.hidden_size = hidden_size

        # --- 2. Load Pretrained Backbone ---
        print(f"Loading pretrained weights from {pretrained_model}...")
        config = Wav2Vec2ConformerConfig(
            num_hidden_layers=self.n_layers,
            hidden_dropout=self.input_dropout,
            attention_dropout=self.input_dropout,
            feat_proj_dropout=self.input_dropout,
            hidden_size=self.hidden_size,
            num_attention_heads=self.n_attention_heads,
            intermediate_size=self.intermediate_size
        )
        self.backbone = Wav2Vec2ConformerModel(config)
        
        # Delete audio front-end
        del self.backbone.feature_extractor
        del self.backbone.feature_projection
        
        # --- 3. Neural Connector ---
        # Calculate input size after patching
        effective_input_dim = input_dim * patch_size
        model_hidden_size = self.backbone.config.hidden_size 
        
        self.neural_projection = nn.Sequential(
            nn.Linear(effective_input_dim, model_hidden_size),
            nn.LayerNorm(model_hidden_size),
            nn.Dropout(self.backbone.config.hidden_dropout)
        )

        # --- 4. Output Head ---
        self.lm_head = nn.Linear(model_hidden_size, num_classes)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x, day_idxs, attention_mask=None):
        """
        neural_data: (Batch, Time, Channels)
        day_idxs: List or Tensor of shape (Batch,) indicating which day each sample belongs to
        """
        # --- 0. Safety Check for Dimensions ---
        # Sometimes dataloaders add a singleton dimension (B, 1, T, C)
        if x.dim() == 4:
            x = x.squeeze(1) 

        # --- A. Apply Day-Specific Transformation ---
        # 1. Gather weights (Batch, Channels, Channels)
        batch_day_weights = torch.stack([self.day_weights[i] for i in day_idxs], dim=0)
        
        # 2. Gather biases (Batch, 1, Channels)
        # REMOVED the .unsqueeze(1) that was causing the 4D crash. 
        # Stacking (1, C) already results in (B, 1, C), which is correct.
        batch_day_biases = torch.stack([self.day_biases[i] for i in day_idxs], dim=0)
        
        # 3. Apply linear transformation: xW + b
        # einsum 'btd' (batch, time, dim) * 'bdk' (batch, dim, output_dim) -> 'btk'
        x = torch.einsum("btd,bdk->btk", x, batch_day_weights) + batch_day_biases
        
        # 4. Activation & Dropout
        x = self.day_layer_activation(x)
        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        # --- B. Patching (Using Unfold for correct Stride) ---
        if self.patch_size > 1:
            # Use unfold to handle patch_stride != patch_size
            # Input x: (B, T, C)
            # Output unfold: (B, N_patches, C, Patch_Size)
            x = x.unfold(dimension=1, size=self.patch_size, step=self.patch_stride)
            
            # We want (B, N_patches, Patch_Size * C)
            # Permute to (B, N, Patch_Size, C) so we can flatten the last two dims
            x = x.permute(0, 1, 3, 2)
            x = x.reshape(x.size(0), x.size(1), -1)

            # Update attention mask to match new time dimension
            if attention_mask is not None:
                # Unfold mask: (B, N_patches, Patch_Size)
                attention_mask = attention_mask.unfold(1, self.patch_size, self.patch_stride)
                # If any value in the patch is valid (1), the patch is valid.
                attention_mask, _ = attention_mask.max(dim=-1)

        # --- C. Projection & Backbone ---
        x = self.neural_projection(x)
        
        outputs = self.backbone.encoder(
            hidden_states=x,
            attention_mask=attention_mask
        )
        
        # --- D. Output ---
        logits = self.lm_head(outputs.last_hidden_state)
        
        return logits
