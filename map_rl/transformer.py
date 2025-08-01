import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import rotary_pe_3d  
import math

def init_weights_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class TransformerLayer(nn.Module):
    def __init__(
        self, 
        d_model=256, 
        n_heads=8, 
        dim_feedforward=1024, 
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Q, K, V projection layers
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection after attention
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Dropouts
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Activation
        self.activation = F.gelu

    def forward(
        self, 
        src: torch.Tensor,             # (B, S, d_model)
        coords_src: torch.Tensor = None,  # (B, S, 3) or None
        causal_mask=None,
        src_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # src shape: (B, S, d_model)
        B, S, _ = src.shape
        
        # 1) Q, K, V projections
        q = self.W_q(src)  # (B, S, d_model)
        k = self.W_k(src)
        v = self.W_v(src)
        
        # 2) Reshape and transpose for multi-head
        # => (B, n_heads, S, head_dim)
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 3) Apply RoPE if coords_src is provided
        if coords_src is not None:
            q = rotary_pe_3d(q, coords_src)
            k = rotary_pe_3d(k, coords_src)
            # v is often unchanged in RoPE
        
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    
        if src_padding_mask is not None:
            key_mask = src_padding_mask.unsqueeze(1).unsqueeze(2)
            scores   = scores.masked_fill(key_mask, float("-inf"))
    
        attn = torch.matmul(F.softmax(scores, -1), v)
        attn = attn.transpose(1, 2).contiguous().view(B, S, self.d_model)
        src2 = self.norm1(src + self.dropout_attn(self.out_proj(attn)))
        ff = self.linear2(self.activation(self.linear1(src2)))
        out = self.norm2(src2 + self.dropout_ff(ff))
        
        if src_padding_mask is not None:
            out = out.masked_fill(src_padding_mask.unsqueeze(-1), 0.0)
        
        return out

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim=768,
        hidden_dim=1024,
        num_layers=4,
        num_heads=8,
    ):
        super().__init__()
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=input_dim,
                n_heads=num_heads,
                dim_feedforward=hidden_dim
            )
            for _ in range(num_layers)
        ])
                
        self.apply(init_weights_kaiming)
               
    def forward(
        self,
        token: torch.Tensor,   # [B, N, input_dim]
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        
        # Use token directly as src
        src = token  # (B, S, D)

        # Use coords directly as coords_src if provided
        coords_src = coords  # (B, S, 3) or None
        
        # Pass through transformer layers
        for layer in self.layers:
            src = layer(
                src=src,
                coords_src=coords_src,
                causal_mask=None,  # no causal masking needed
            )

        return src

class ActionTransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
        action_dim: int,
        action_pred_horizon: int = 1,
    ):
        super().__init__()
        
        self.query_embed = nn.Embedding(action_pred_horizon, d_model)  # [3, d_model]
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu"
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.action_pred_horizon = action_pred_horizon
        
    def forward(self, memory, state) -> torch.Tensor:

        # state # [B, 1, d_model]

        B, N, d_model = memory.shape
  
        # memory = memory.view(B, fs*N, d_model)             # [B, 2*N, d_model]
        memory = memory.permute(1, 0, 2).contiguous()     # [N, B, d_model]
        
        query_pos = self.query_embed.weight                # [3, d_model]
        query_pos = query_pos.unsqueeze(1).repeat(1, B, 1) # [3, B, d_model]
        
        state = state.permute(1, 0, 2).contiguous()
        tgt = torch.cat([state, query_pos], dim=0)
        
        decoder_out = self.decoder(
            tgt=tgt,    # [T, B, d_model]
            memory=memory,      # [N, B, d_model
        ) 
        
        decoder_out = decoder_out.permute(1, 0, 2)         # [B, 4, d_model]
        return decoder_out[:, 1:, :]

class StateProj(nn.Module):
    def __init__(self, state_dim=42, output_dim=768):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(state_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim)
        )
        
        self.apply(init_weights_kaiming)

    def forward(self, state):
        out = self.mlp(state)
        return out
