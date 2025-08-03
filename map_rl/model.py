import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import gymnasium as gym
from typing import Optional, List

import torchvision.models as models
from torchvision.models import ResNet18_Weights

from utils import transform, get_3d_coordinates


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal layer initialization matching CleanRL PPO defaults."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class DictArray(object):
    """Utility wrapper that stores a dict of torch tensors with the same leading buffer shape."""

    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    dtype = (
                        torch.float32 if v.dtype in (np.float32, np.float64) else
                        torch.uint8 if v.dtype == np.uint8 else
                        torch.int16 if v.dtype == np.int16 else
                        torch.int32 if v.dtype == np.int32 else
                        torch.bool if v.dtype in (np.bool_, bool) else
                        v.dtype
                    )
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {k: v[index] for k, v in self.data.items()}

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k, v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)


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
    def __init__(self, d_model, n_heads, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_padding_mask=None, **kwargs):
        src2 = self.self_attn(src, src, src, key_padding_mask=src_padding_mask, need_weights=False)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class LocalFeatureFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        ff_mult: int = 4,
        radius: float = 0.06,
        k: int = 8,
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
        
        # key-padding mask for attention (True → ignore)
        pad_mask = torch.cat(
            [
                torch.zeros_like(invalid[..., :1]),  # query token (#0) is always valid
                invalid
            ],
            dim=-1
        ).view(B * N, self.k + 1)                    # (B*N, k+1)

        # reshape to (B*N, S, C) for the transformer layer
        BM = B * N
        fused = self.attn(
            tokens.view(BM, self.k + 1, C).contiguous(),
            src_padding_mask=pad_mask,
        )                                            # (BM, k+1, C)

        # return only the query position (index 0 within each group)
        fused_q = fused[:, 0, :].view(B, N, C) + q_feat
        # fused_q = fused[:, 0, :].view(B, N, C) 
        
        return fused_q

class MapAwareFeatureExtractor(nn.Module):
    def __init__(self, sample_obs, decoder: Optional[nn.Module] = None):
        """
        sample_obs : dict
            A single observation sample (used only for inferring tensor shapes).
        decoder : nn.Module or None
            Voxel-feature decoder.  If None, the whole map branch is skipped.
        """
        super().__init__()
        self.decoder = decoder     # None: RGB-only mode

        feature_size = 256
        self.out_features = 0

        # ------------------------------------------------------------------ CNN -
        # Pre-trained ResNet-18 up to the last conv block (512-D feature map).
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.cnn                    = nn.Sequential(*list(resnet.children())[:-2])
        self.rgb_feature_dim        = 512    # channel dimension after layer4

        with torch.no_grad():
            dummy           = sample_obs["rgb"].float().permute(0, 3, 1, 2) / 255.0
            cnn_out_shape   = self.cnn(dummy.cpu()).shape          # (B, 512, H', W')
            n_flatten       = self.rgb_feature_dim * cnn_out_shape[2] * cnn_out_shape[3]

        # Global RGB head (always available – even if map is disabled).
        self.rgb_proj = nn.Sequential(
            nn.Flatten(), nn.Linear(n_flatten, feature_size), nn.ReLU()
        )
        self.out_features += feature_size

        # ---------------------------------------------------------------- Map 3-D -
        if self.decoder is not None:                 # build map-specific modules
            map_feature_dim          = 3 + 768
            self.map_encoder         = PointNet(map_feature_dim, 256)
            self.local_fusion        = LocalFeatureFusion(dim=self.rgb_feature_dim, k=8)
            self.map_feature_proj    = nn.Linear(768, self.rgb_feature_dim)

            fused_flat               = self.rgb_feature_dim * cnn_out_shape[2] * cnn_out_shape[3]
            self.fused_proj          = nn.Sequential(
                nn.Flatten(), nn.Linear(fused_flat, feature_size), nn.ReLU()
            )

            self.out_features       += 256          # local fusion
            self.out_features       += 256          # global PointNet

        # ---------------------------------------------------------------- State --
        self.state_proj = None
        if "state" in sample_obs:
            state_size      = sample_obs["state"].shape[-1]
            self.state_proj = nn.Linear(state_size, 256)
            self.out_features += 256

    # --------------------------------------------------------------------------- #
    #  forward                                                                    #
    # --------------------------------------------------------------------------- #
    def forward(
        self,
        observations: dict,
        map_features: Optional[List] = None,
        *,
        use_map: bool = True,
    ) -> torch.Tensor:
        """
        observations : dict
            Batched observations containing at least 'rgb'.
        map_features : list or None
            Per-environment voxel grids (may be None).
        use_map : bool, default=True
            If False, skip ALL map-related computation even when map_features
            are provided.  Useful for ablations.
        """
        B        = observations["rgb"].size(0)
        encoded  = []                               # list[(B, D_i)]

        # ----------------------------------------------------------- RGB backbone
        rgb      = observations["rgb"].float().permute(0, 3, 1, 2) / 255.0
        rgb_fmap = self.cnn(rgb)                    # (B, 2048, H', W')
        rgb_vec  = self.rgb_proj(rgb_fmap)          # (B, 256)

        # Fast path: RGB-only ----------------------------------------------------
        if not use_map or self.decoder is None or map_features is None:
            encoded.append(rgb_vec)

        # Map-aware path ---------------------------------------------------------
        else:
            # 1) Gather voxel coords & raw features ------------------------------
            coords_batch, raw_batch = [], []
            for g in map_features:                           # Python loop per env
                coords_batch.append(g.levels[1].coords)      # (L_i, 3)
                raw_batch.append(g.query_voxel_feature(coords_batch[-1]))  # (L_i, F_raw)

            # 2) Decode all voxels in a single call ------------------------------
            with torch.no_grad():
                dec_cat = self.decoder(torch.cat(raw_batch, dim=0))          # (ΣL, 768)

            dec_split = dec_cat.split([c.size(0) for c in coords_batch], dim=0)

            # 3) Pad for attention ----------------------------------------------
            kv_xyz = torch.nn.utils.rnn.pad_sequence(coords_batch, batch_first=True)  # (B,Lmax,3)
            kv_raw = torch.nn.utils.rnn.pad_sequence(dec_split,  batch_first=True)    # (B,Lmax,768)

            Lmax   = kv_xyz.size(1)
            pad_mask = torch.arange(Lmax, device=kv_xyz.device).expand(B, -1)
            pad_mask = pad_mask >= torch.tensor([c.size(0) for c in coords_batch], device=kv_xyz.device).unsqueeze(1)    # (B,Lmax)
            kv_feat = self.map_feature_proj(kv_raw)                                   # (B,Lmax,2048)

            # 4) Build query coords + features -----------------------------------
            depth   = observations["depth"].permute(0, 3, 1, 2).float() / 1000.0
            pose    = observations["sensor_param"]["base_camera"]["extrinsic_cv"]

            Hf, Wf  = rgb_fmap.shape[2:]
            depth_s = F.interpolate(depth, size=(Hf, Wf), mode="nearest-exact")

            fx = fy = cx = cy = 64          # TODO: replace with real intrinsics
            q_xyz, _ = get_3d_coordinates(
                depth_s, pose, fx=fx, fy=fy, cx=cx, cy=cy,
                original_size=observations["rgb"].shape[1:3],
            )
            q_xyz  = q_xyz.permute(0, 2, 3, 1).reshape(B, -1, 3)              # (B,H'W',3)
            q_feat = rgb_fmap.permute(0, 2, 3, 1).reshape(B, -1, self.rgb_feature_dim)

            # 5) Local 3-D ↔ 2-D fusion ----------------------------------------
            fused  = self.local_fusion(q_xyz, q_feat, kv_xyz, kv_feat, pad_mask)  # (B,H'W',2048)
            fused  = fused.reshape(B, self.rgb_feature_dim, Hf, Wf)
            fused_vec = self.fused_proj(fused)                                # (B,256)

            # 6) Global PointNet encoding ---------------------------------------
            concat   = [torch.cat([c, d], dim=-1) for c, d in zip(coords_batch, dec_split)]
            pad_3d   = torch.nn.utils.rnn.pad_sequence(concat, batch_first=True)  # (B,Lmax,771)
            map_vec  = self.map_encoder(pad_3d)                                  # (B,256)

            # Collect outputs ----------------------------------------------------
            encoded.extend([fused_vec, map_vec])

        # -------------------------------------------------------------- State MLP
        if self.state_proj is not None:
            encoded.append(self.state_proj(observations["state"]))

        # ------------------------------------------------------------ Final concat
        return torch.cat(encoded, dim=1)                # (B, self.out_features)


class Agent(nn.Module):
    """Actor-Critic network (Gaussian continuous actions) built on NatureCNN features."""

    def __init__(self, envs, sample_obs, decoder: Optional[nn.Module] = None):
        super().__init__()
        self.feature_net = MapAwareFeatureExtractor(sample_obs=sample_obs, decoder=decoder)
        # self.feature_net = NatureCNN(sample_obs=sample_obs)
        latent_size = 768
        
        # Critic
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )

        # Actor: mean of Gaussian policy
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(
                nn.Linear(512, int(np.prod(envs.unwrapped.single_action_space.shape))),
                std=0.01 * np.sqrt(2),
            ),
        )
        # Log-std is state-independent
        self.actor_logstd = nn.Parameter(
            torch.ones(1, int(np.prod(envs.unwrapped.single_action_space.shape))) * -0.5
        )

    # Convenience helpers -----------------------------------------------------

    def get_features(self, x, map_features=None):
        return self.feature_net(x, map_features=map_features)

    def get_value(self, x, map_features=None):
        x = self.get_features(x, map_features=map_features)
        return self.critic(x)

    def get_action(self, x, map_features=None, deterministic: bool = False):
        x = self.get_features(x, map_features=map_features)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x, map_features=None, action=None):
        x = self.get_features(x, map_features=map_features)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )
