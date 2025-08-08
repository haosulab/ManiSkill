import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import gymnasium as gym
from typing import Optional, List
import xformers.ops as xops
from model.module import PointNet, LocalFeatureFusion, layer_init
from utils.operator import get_3d_coordinates
from model.vision_encoder import DINO2DFeatureEncoder, PlainCNNFeatureEncoder

class FeatureExtractor(nn.Module):
    def __init__(
            self,
            sample_obs: dict,
            feature_size: int = 256,
            vision_encoder: str = "plain_cnn",
            decoder: Optional[nn.Module] = None,
            use_map: bool = True,
            use_local_fusion: bool = False,
        ) -> None:
        super().__init__()
        
        # --------------------------------------------------------------- Vision
        object.__setattr__(self, "_decoder", decoder)  # None → RGB-only mode
        
        if vision_encoder == 'dino':
            self.vision_encoder = DINO2DFeatureEncoder(embed_dim=64)
            n_flatten = 36 * self.vision_encoder.embed_dim # 36 = 6 * 6
        elif vision_encoder == 'plain_cnn':
            self.vision_encoder = PlainCNNFeatureEncoder(embed_dim=64)
            n_flatten = 36 * self.vision_encoder.embed_dim # 36 = 6 * 6
        else:
            raise ValueError(f"Vision encoder {vision_encoder} not supported")
        
        self.global_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Linear(2048, feature_size),
            nn.LayerNorm(feature_size),
            nn.ReLU(),
        )
        
        # --------------------------------------------------------------- Map 3-D
        self.use_map = use_map and (decoder is not None)
        self.use_local_fusion = self.use_map and use_local_fusion

        if self.use_map:
            map_raw_dim = 768  # output dim of *decoder*
            self.map_encoder = PointNet(3 + map_raw_dim, feature_size)

            if self.use_local_fusion:
                self.map_feature_proj = nn.Linear(map_raw_dim, self.vision_encoder.embed_dim)
                self.local_fusion = LocalFeatureFusion(dim=self.vision_encoder.embed_dim, k=2)

        # --------------------------------------------------------------- State
        self.state_proj = None
        if "state" in sample_obs:
            state_size = sample_obs["state"].shape[-1]
            self.state_proj = nn.Linear(state_size, feature_size)

        num_branches = 1  # state
        num_branches += 1  # global RGB 
        if self.use_map:
            num_branches += 1  # global map
        self.output_dim = num_branches * feature_size

    def _local_fusion(
        self,
        observations: dict,
        image_fmap: torch.Tensor,
        coords_batch: List[torch.Tensor],
        dec_split: List[torch.Tensor],
    ) -> torch.Tensor:
        B = image_fmap.size(0)

        kv_xyz = torch.nn.utils.rnn.pad_sequence(coords_batch, batch_first=True)  # (B,Lmax,3)
        kv_raw = torch.nn.utils.rnn.pad_sequence(dec_split, batch_first=True)     # (B,Lmax,768)

        Lmax = kv_xyz.size(1)
        pad_mask = torch.arange(Lmax, device=kv_xyz.device).expand(B, -1)
        pad_mask = pad_mask >= torch.tensor(
            [c.size(0) for c in coords_batch], device=kv_xyz.device
        ).unsqueeze(1)  # (B,Lmax)

        kv_feat = self.map_feature_proj(kv_raw)  # (B,Lmax,embed_dim)

        depth = observations["depth"].permute(0, 3, 1, 2).float() / 1000.0
        pose = observations["sensor_param"]["base_camera"]["extrinsic_cv"]

        Hf = Wf = 6
        depth_s = F.interpolate(depth, size=(Hf, Wf), mode="nearest-exact")

        fx = fy = cx = cy = 42  # TODO: replace with real intrinsics
        q_xyz, _ = get_3d_coordinates(
            depth_s,
            pose,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            original_size=(84, 84),
        )  # B, H, W, 3

        q_xyz = q_xyz.permute(0, 2, 3, 1).reshape(B, -1, 3)            # (B, N, 3)
        q_feat = image_fmap.permute(0, 2, 3, 1).reshape(B, -1, self.vision_encoder.embed_dim)     # (B, N, C)

        fused = self.local_fusion(q_xyz, q_feat, kv_xyz, kv_feat, pad_mask)  # (B, N, C)
        image_fmap = fused.permute(0, 2, 1).reshape(
            B, self.vision_encoder.embed_dim, Hf, Wf
        )  # (B, C, Hf, Wf)

        return image_fmap

    def forward(
        self,
        observations: dict,
        map_features: Optional[List] = None,
    ) -> torch.Tensor:
    
        B = observations["rgb"].size(0)
        encoded: List[torch.Tensor] = []

        if self.state_proj is not None:
            encoded.append(self.state_proj(observations["state"]))

        image = observations["rgb"].float().permute(0, 3, 1, 2) / 255.0
        image_fmap = self.vision_encoder(image) # B, C, Hf, Wf

        if not self.use_map:
            image_vec = self.global_proj(image_fmap) 
            encoded.append(image_vec)
            return torch.cat(encoded, dim=1)
        
        assert map_features is not None, "map_features must be provided when use_map=True"

        coords_batch, raw_batch = [], []
        for g in map_features:
            coords = g.levels[1].coords
            coords_batch.append(coords)
            raw_batch.append(g.query_voxel_feature(coords))

        with torch.no_grad():
            dec_cat = self._decoder(torch.cat(raw_batch, dim=0))          # (ΣL, 768)

        dec_split = dec_cat.split([c.size(0) for c in coords_batch], dim=0)

        concat = [torch.cat([c, d], dim=-1) for c, d in zip(coords_batch, dec_split)]
        pad_3d = torch.nn.utils.rnn.pad_sequence(concat, batch_first=True)  # (B, Lmax, 771)
        map_vec = self.map_encoder(pad_3d)  # (B, 256)
        encoded.append(map_vec)

        if self.use_local_fusion:
            image_fmap = self._local_fusion(
                observations=observations,
                image_fmap=image_fmap,
                coords_batch=coords_batch,
                dec_split=dec_split,
            )
        
        image_vec = self.global_proj(image_fmap)                                # B, 256
        encoded.append(image_vec)

        return torch.cat(encoded, dim=1)

class Agent(nn.Module):
    """Actor-Critic network (Gaussian continuous actions) built on NatureCNN features."""

    def __init__(
        self, 
        envs, 
        sample_obs, 
        decoder: Optional[nn.Module] = None, 
        use_map: bool = True, 
        use_local_fusion: bool = False, 
        vision_encoder: str = "plain_cnn"
    ):
        super().__init__()
        self.feature_net = FeatureExtractor(
            sample_obs=sample_obs, 
            decoder=decoder, 
            use_map=use_map, 
            use_local_fusion=use_local_fusion, 
            vision_encoder=vision_encoder
        )
        latent_size = self.feature_net.output_dim
        
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