import torch
import torch.nn as nn
import torch.nn.functional as F
# try:
#     import xformers.ops as xops
#     HAS_XFORMERS = True
# except ImportError:
#     HAS_XFORMERS = False
HAS_XFORMERS = False

from typing import Optional
from utils import rotary_pe_3d
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal layer initialization matching CleanRL PPO defaults."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PointNet(nn.Module):
    def __init__(self, input_dim, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(input_dim, 256)),
            nn.LayerNorm(256),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.LayerNorm(256),
            nn.ReLU(),
            layer_init(nn.Linear(256, output_dim)),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.net(x)
        return torch.max(x, dim=1)[0]

class TransformerLayer(nn.Module):
    def __init__(
        self, 
        d_model=256, 
        n_heads=8, 
        dim_feedforward=1024, 
        dropout=0.1,
        use_xformers: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_xformers = use_xformers and HAS_XFORMERS

        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.activation = F.gelu

    def forward(
        self, 
        src: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        coords_src: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, _ = src.shape
        
        q = self.W_q(src).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(src).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(src).view(B, S, self.n_heads, self.head_dim)

        if coords_src is not None:
            q = rotary_pe_3d(q, coords_src)
            k = rotary_pe_3d(k, coords_src)
        
        if self.use_xformers:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            
            attn_bias = None
            if key_padding_mask is not None:
                # xformers >=0.0.23 removed `KeyPaddingMask`. Build a dense bias tensor instead.
                seq_len = key_padding_mask.size(1)
                mask = key_padding_mask[:, None, None, :].to(q.dtype)
                attn_bias = mask.expand(-1, self.n_heads, seq_len, -1) * (-1e9)
            else:
                attn_bias = None
                
            attn = xops.memory_efficient_attention(
                q, k, v,
                attn_bias=attn_bias,
                p=self.dropout_attn.p if self.training else 0.0,
            )  # (B, S, H, D)    

        else:
            # PyTorch's scaled_dot_product_attention expects (B, n_heads, S, head_dim)
            v = v.transpose(1, 2).contiguous()
            # Build an attention mask from the key\_padding\_mask (True → ignore)
            attn_mask = None
            if key_padding_mask is not None:
                # expected shape: (B, 1, 1, K) broadcastable to (B, H, Q, K)
                attn_mask = key_padding_mask[:, None, None, :].to(torch.bool)
            
            attn = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_attn.p if self.training else 0.0,
            )
            attn = attn.transpose(1, 2).contiguous() # (B, S, H, D)
        
        # Collapse heads ---------------------------------------------------
        attn = attn.reshape(B, S, self.d_model).contiguous()

        # Residual & FF -----------------------------------------------------
        src2 = self.norm1(src + self.dropout_attn(self.out_proj(attn)))
        ff = self.linear2(self.activation(self.linear1(src2)))
        out = self.norm2(src2 + self.dropout_ff(ff))
        return out

class LocalFeatureFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        ff_mult: int = 4,
        radius: float = 0.06,
        k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.radius, self.k = radius, k
        self.attn = TransformerLayer(
            d_model=dim,
            n_heads=n_heads,
            dim_feedforward=dim * ff_mult,
            dropout=dropout,
        )

    # ----------------------------------------------------------
    # Find neighbor indices within <radius>; pad with query itself
    # ----------------------------------------------------------
    def _neigh_indices(
        self,
        q_xyz: torch.Tensor,           # (B, N, 3)  – query coordinates
        kv_xyz: torch.Tensor,          # (B, L, 3)  – scene coordinates
        kv_pad: Optional[torch.Tensor] # (B, L) bool – True → padding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        idx     : (B, N, k) long  – neighbor indices (query-padded)
        invalid : (B, N, k) bool  – True → padding slot
        """
        dist = torch.cdist(q_xyz, kv_xyz)                      # (B, N, L)
        if kv_pad is not None:
            dist = dist.masked_fill(kv_pad[:, None, :], float("inf"))

        # keep only points ≤ radius
        dist = torch.where(dist <= self.radius, dist, float("inf"))
        k = self.k

        # 1) take top-k closest (up to k). If fewer, remaining are arbitrary for now.
        _, idx_topk = dist.topk(k, largest=False, dim=-1)      # (B, N, k)

        # 2) mark invalid (padding) slots
        gather_dist = dist.gather(-1, idx_topk)                # (B, N, k)
        invalid = gather_dist.isinf()                          # True → padding slot

        # 3) overwrite padding slots with dummy index 0 (will be replaced by query itself)
        query_idx = torch.zeros_like(idx_topk)                 # value 0 is arbitrary
        idx = torch.where(invalid, query_idx, idx_topk)        # (B, N, k)

        return idx, invalid

    # ----------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------
    def forward(
        self,
        q_xyz:   torch.Tensor,                # (B, N, 3)
        q_feat:  torch.Tensor,                # (B, N, C)
        kv_xyz:  torch.Tensor,                # (B, L, 3)
        kv_feat: torch.Tensor,                # (B, L, C)
        kv_pad:  Optional[torch.Tensor] = None  # (B, L) bool
    ) -> torch.Tensor:
        B, N, C = q_feat.shape
        idx, invalid = self._neigh_indices(q_xyz, kv_xyz, kv_pad)  # (B, N, k)

        # Debug        
        # num_valid = (~invalid).sum()
        # print(f"Number of valid neighbors: {num_valid.item()}")

        # gather neighbor coordinates / features
        batch = torch.arange(B, device=q_feat.device).view(B, 1, 1)
        neigh_xyz  = kv_xyz[batch.expand_as(idx), idx]             # (B, N, k, 3)
        neigh_feat = kv_feat[batch.expand_as(idx), idx]            # (B, N, k, C)

        # replace padding slots with the query point itself
        neigh_xyz [invalid] = q_xyz.unsqueeze(2).expand(-1, -1, self.k, -1)[invalid]
        neigh_feat[invalid] = q_feat.unsqueeze(2).expand(-1, -1, self.k, -1)[invalid]

        # concatenate query token with neighbor tokens
        tokens = torch.cat([q_feat.unsqueeze(2), neigh_feat], dim=2)  # (B, N, k+1, C)
        # token_xyz = torch.cat([q_xyz.unsqueeze(2), neigh_xyz], dim=2)  # (B, N, k+1, 3)
        
        # key-padding mask for attention (True → ignore)
        key_padding_mask = torch.cat(
            [torch.zeros_like(invalid[..., :1]), invalid], dim=-1
        ).view(B * N, self.k + 1)

        # reshape to (B*N, S, C) for the transformer layer
        BM = B * N
        fused = self.attn(
            tokens.view(BM, self.k + 1, C).contiguous(),
            key_padding_mask=key_padding_mask,
        )  # (BM, k+1, C)

        # return only the query position (index 0 within each group)
        fused_q = fused[:, 0, :].view(B, N, C) + q_feat
        # fused_q = fused[:, 0, :].view(B, N, C) 
        
        return fused_q