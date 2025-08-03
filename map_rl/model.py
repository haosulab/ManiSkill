import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import gymnasium as gym
from typing import Optional, List

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
        super().__init__()
        self.decoder = decoder
        if self.decoder:
            self.decoder.eval()
            for p in self.decoder.parameters():
                p.requires_grad = False

        extractors = {}
        self.out_features = 0
        feature_size = 256

        # Nature CNN for RGB
        in_channels = sample_obs["rgb"].shape[-1]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )
        with torch.no_grad():
            sample_rgb_permuted = sample_obs["rgb"].float().permute(0, 3, 1, 2).cpu()
            cnn_out_shape = self.cnn(sample_rgb_permuted).shape
            self.rgb_feature_dim = cnn_out_shape[1]
            n_flatten = self.rgb_feature_dim * cnn_out_shape[2] * cnn_out_shape[3]
            fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        
        extractors["rgb"] = nn.Sequential(self.cnn, nn.Flatten(), fc)
        self.out_features += feature_size

        # State MLP
        if "state" in sample_obs:
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

        # Map Feature Encoder (PointNet)
        if self.decoder:
            map_feature_dim = 3 + 768 
            self.map_encoder = PointNet(input_dim=map_feature_dim, output_dim=256)
            self.out_features += 256
            self.local_fusion = LocalFeatureFusion(dim=self.rgb_feature_dim, k=8)
            self.map_feature_proj = nn.Linear(768, self.rgb_feature_dim)

            with torch.no_grad():
                fused_feature_dim = self.rgb_feature_dim * cnn_out_shape[2] * cnn_out_shape[3]

            self.feature_proj = nn.Sequential(
                nn.Flatten(),
                nn.Linear(fused_feature_dim, feature_size),
                nn.ReLU()
            )


    # --------------------------------------------------------------------------- #
    #  MapAwareFeatureExtractor.forward                                           #
    # --------------------------------------------------------------------------- #
    def forward(self, observations, map_features: Optional[List] = None) -> torch.Tensor:
        B          = observations["rgb"].size(0)    
        encoded    = []                                           # list of (B, D_i)

        # ------------------------------------------------------------------ RGB -
        rgb   = observations["rgb"].float().permute(0, 3, 1, 2) / 255.0
        rgb_f = self.cnn(rgb)                                     # (B, C_rgb, H', W')

        # ---------------------------------------------------------------- Map 3-D -
        if map_features is not None and self.decoder is not None:
            # -------- 1. Gather per-grid coords + raw voxel features -------------
            coords_list, rawfeat_list = [], []
            for g in map_features:                                # one Python pass
                coords            = g.levels[1].coords            # (Lᵢ, 3)
                rawfeat           = g.query_voxel_feature(coords) # (Lᵢ, F_raw)
                coords_list.append(coords)
                rawfeat_list.append(rawfeat)

            # -------- 2. Decode ALL voxel features in one call -------------------
            with torch.no_grad():
                raw_cat      = torch.cat(rawfeat_list, dim=0)         # (ΣLᵢ, F_raw)
                dec_cat      = self.decoder(raw_cat)                  # (ΣLᵢ, 768)
            
            split_sizes  = [c.size(0) for c in coords_list]
            dec_list     = dec_cat.split(split_sizes, dim=0)      # list[Lᵢ, 768]

            # -------- 3. Pad for local fusion attention --------------------------
            kv_xyz       = torch.nn.utils.rnn.pad_sequence(coords_list, batch_first=True)  # (B,Lmax,3)
            kv_raw       = torch.nn.utils.rnn.pad_sequence(dec_list,   batch_first=True)  # (B,Lmax,768)

            # key-padding mask: True ➜ ignore
            lens         = torch.tensor(split_sizes, device=kv_xyz.device)
            idx_row      = torch.arange(kv_xyz.size(1), device=kv_xyz.device).expand(B, -1)
            kv_pad       = idx_row >= lens.unsqueeze(1)                                 # (B,Lmax)
            kv_feat      = self.map_feature_proj(kv_raw)                                # (B,Lmax,C_rgb)

            # -------- 4. Build query (pixel) 3-D coords + features ----------------
            depth   = observations["depth"].permute(0, 3, 1, 2).float() / 1000.0
            pose    = observations["sensor_param"]["base_camera"]["extrinsic_cv"]
            Hf, Wf  = rgb_f.shape[2:]
            depth_s = F.interpolate(depth, size=(Hf, Wf), mode="nearest-exact")

            fx = fy = cx = cy = 64                           # TODO: replace with real intrinsics
            q_xyz, _ = get_3d_coordinates(
                depth_s, pose, fx=fx, fy=fy, cx=cx, cy=cy,
                original_size=observations["rgb"].shape[1:3],
            )
            q_xyz  = q_xyz.permute(0, 2, 3, 1).reshape(B, -1, 3)                # (B,H'W',3)
            q_feat = rgb_f.permute(0, 2, 3, 1).reshape(B, -1, self.rgb_feature_dim)  # (B,H'W',C_rgb)

            # -------- 5. Local fusion (3-D ↔ 2-D attention) -----------------------
            fused  = self.local_fusion(q_xyz, q_feat, kv_xyz, kv_feat, kv_pad)   # (B,H'W',C_rgb)
            fused  = fused.reshape(B, Hf, Wf, -1).permute(0, 3, 1, 2)            # (B,C_rgb,H',W')
            proj   = self.feature_proj(fused)                                    # (B,256)
            encoded.append(proj)

            # -------- 6. Global map encoding via PointNet ------------------------
            concat_list = [torch.cat([c, d], dim=-1) for c, d in zip(coords_list, dec_list)]  # (Lᵢ, 3+768)
            pad_3d      = torch.nn.utils.rnn.pad_sequence(concat_list, batch_first=True)      # (B,Lmax, 771)
            map_enc     = self.map_encoder(pad_3d)                                # (B,256)
            encoded.append(map_enc)

        else:
            # No map features → fallback to standard RGB extractor
            encoded.append(self.extractors["rgb"](rgb))

        # ------------------------------------------------------------- State MLP -
        if "state" in self.extractors:
            encoded.append(self.extractors["state"](observations["state"]))

        # ----------------------------------------------------------- Final concat -
        return torch.cat(encoded, dim=1)                 # (B, self.out_features)


class NatureCNN(nn.Module):
    def __init__(self, sample_obs):
        super().__init__()

        extractors = {}

        self.out_features = 0
        feature_size = 256
        in_channels=sample_obs["rgb"].shape[-1]
        image_size=(sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])


        # here we use a NatureCNN architecture to process images, but any architecture is permissble here
        cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=8,
                stride=4,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

        # to easily figure out the dimensions after flattening, we pass a test tensor
        with torch.no_grad():
            n_flatten = cnn(sample_obs["rgb"].float().permute(0,3,1,2).cpu()).shape[1]
            fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        extractors["rgb"] = nn.Sequential(cnn, fc)
        self.out_features += feature_size

        if "state" in sample_obs:
            # for state data we simply pass it through a single linear layer
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations, map_features: Optional[List] = None) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                obs = obs.float().permute(0,3,1,2)
                obs = obs / 255
            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)

class CNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cnn =  nn.Sequential(
            # 128 → 64
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            # 64 → 32
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 32 → 16
            nn.Conv2d(256, 768, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )                       
    def forward(self, x):
        return self.cnn(x)

class Agent(nn.Module):
    """Actor-Critic network (Gaussian continuous actions) built on NatureCNN features."""

    def __init__(self, envs, sample_obs, decoder: Optional[nn.Module] = None):
        super().__init__()
        self.feature_net = MapAwareFeatureExtractor(sample_obs=sample_obs, decoder=decoder)
        # self.feature_net = NatureCNN(sample_obs=sample_obs)
        latent_size = self.feature_net.out_features
        
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
