import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import gymnasium as gym
from typing import Optional, List
import xformers.ops as xops
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from module import PointNet, LocalFeatureFusion, layer_init
from utils import get_3d_coordinates, transform, get_visual_features_dino

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

class MapAwareFeatureExtractor(nn.Module):
    def __init__(self, sample_obs, decoder: Optional[nn.Module] = None):
        """
        sample_obs : dict
            A single observation sample (used only for inferring tensor shapes).
        decoder : nn.Module or None
            Voxel-feature decoder.  If None, the whole map branch is skipped.
        """
        super().__init__()
        # Store decoder as a plain attribute so its parameters are not registered
        object.__setattr__(self, "_decoder", decoder)  # None → RGB-only mode

        feature_size = 256
        self.out_features = 0

        # ------------------------------------------------------------------ CNN -
        # Pre-trained ResNet-18 up to layer2 (128-D feature map).
        dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
        object.__setattr__(self, "backbone", dino)

        # with torch.no_grad():
        #    x = get_visual_features_dino(self.dino, transform(sample_obs["rgb"].float().permute(0,3,1,2).cpu() / 255.0)) # B, C, Hf, Wf
        # n_flatten = x.shape[1] * x.shape[2] * x.shape[3]
        n_flatten = 256 * 384   

        # resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # self.cnn                    = nn.Sequential(*list(resnet.children())[:6])
        # self.rgb_feature_dim        = 128    # channel dimension after layer2

        # with torch.no_grad():
        #     dummy           = sample_obs["rgb"].float().permute(0, 3, 1, 2) / 255.0
        #     cnn_out_shape   = self.cnn(dummy.cpu()).shape          # (B, 192, H', W')
        #     n_flatten       = self.rgb_feature_dim * cnn_out_shape[2] * cnn_out_shape[3]

        # Global RGB head (always available – even if map is disabled).
        self.fused_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Linear(2048, feature_size),
            nn.LayerNorm(feature_size),
            nn.ReLU()
        )

        self.out_features += feature_size

        # ---------------------------------------------------------------- Map 3-D -
        if self._decoder is not None:                 # build map-specific modules
            map_feature_dim          = 3 + 768
            self.map_encoder         = PointNet(map_feature_dim, feature_size)
            self.local_fusion        = LocalFeatureFusion(dim=self.backbone.embed_dim, k=2)
            self.map_feature_proj    = nn.Linear(768, self.backbone.embed_dim)

            self.out_features       += feature_size          # local fusion
            self.out_features       += feature_size          # global PointNet

        # ---------------------------------------------------------------- State --
        self.state_proj = None
        if "state" in sample_obs:
            state_size      = sample_obs["state"].shape[-1]
            self.state_proj = nn.Linear(state_size, feature_size)
            self.out_features += feature_size

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

        B        = observations["rgb"].size(0)
        encoded  = []                               # list[(B, D_i)]

        # ----------------------------------------------------------- RGB backbone
        rgb      = observations["rgb"].float().permute(0, 3, 1, 2) / 255.0

        with torch.no_grad():
            rgb_fmap = get_visual_features_dino(self.backbone, transform(rgb)) # B, C, Hf, Wf

        # Fast path: RGB-only ----------------------------------------------------
        if not use_map or self._decoder is None or map_features is None:
            rgb_vec  = self.fused_proj(rgb_fmap) # B, N, C -> B, 256
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
                dec_cat = self._decoder(torch.cat(raw_batch, dim=0))          # (ΣL, 768)

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

            Hf = Wf = 16
            depth_s = F.interpolate(depth, size=(Hf, Wf), mode="nearest-exact")

            fx = fy = cx = cy = 112          # TODO: replace with real intrinsics
            q_xyz, _ = get_3d_coordinates(
                depth_s, pose, fx=fx, fy=fy, cx=cx, cy=cy,
                original_size=(224, 224),
            ) # B, H, W, 3

            q_xyz  = q_xyz.permute(0, 2, 3, 1).reshape(B, -1, 3) # B, N, 3   
            q_feat = rgb_fmap.permute(0, 2, 3, 1).reshape(B, -1, 384) # B, N, C

            # 5) Local 3-D ↔ 2-D fusion ----------------------------------------
            fused  = self.local_fusion(q_xyz, q_feat, kv_xyz, kv_feat, pad_mask) # B, N, C
            fused  = fused.permute(0, 2, 1).reshape(B, self.backbone.embed_dim, Hf, Wf) # B, C, Hf, Wf
            fused_vec = self.fused_proj(fused)                                # B, 256

            # 6) Global PointNet encoding ---------------------------------------
            concat   = [torch.cat([c, d], dim=-1) for c, d in zip(coords_batch, dec_split)]
            pad_3d   = torch.nn.utils.rnn.pad_sequence(concat, batch_first=True)  # B, Lmax, 771
            map_vec  = self.map_encoder(pad_3d)                                  # B, 256

            # Collect outputs ----------------------------------------------------
            encoded.extend([fused_vec, map_vec])

        # -------------------------------------------------------------- State ML
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
