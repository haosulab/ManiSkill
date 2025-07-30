import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import gymnasium as gym

from transformer import TransformerEncoder, ActionTransformerDecoder, StateProj
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


class NatureCNN(nn.Module):
    """Nature CNN feature extractor (RGB images + optional state vector)."""

    def __init__(self, sample_obs):
        super().__init__()

        extractors = {}
        self.out_features = 0
        feature_size = 256

        in_channels = sample_obs["rgb"].shape[-1]

        # Nature CNN backbone for image input
        cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = cnn(sample_obs["rgb"].float().permute(0, 3, 1, 2).cpu()).shape[1]
            fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        extractors["rgb"] = nn.Sequential(cnn, fc)
        self.out_features += feature_size

        # Optional low-dimensional state vector
        if "state" in sample_obs:
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                obs = obs.float().permute(0, 3, 1, 2) / 255.0
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


class ActionModel(nn.Module):
    def __init__(self, sample_obs):
        super().__init__()

        self.cnn = CNN(in_channels=sample_obs["rgb"].shape[-1])
        self.state_mlp = StateProj(state_dim=13, output_dim=768)

        self.transformer_encoder = TransformerEncoder(
            input_dim=768,
            hidden_dim=1024,
            num_layers=1,
            num_heads=8,
        )

        self.transformer_decoder = ActionTransformerDecoder(
            d_model=768,
            nhead=8,
            num_decoder_layers=4,
            dim_feedforward=768,
            dropout=0.1,
            action_dim=13,
        )

    def _process_rgb(self, rgb_obs):
        rgb = rgb_obs.permute(0, 3, 1, 2)
        rgb = transform(rgb.float() / 255.0)
        return self.cnn(rgb)

    def _process_state(self, state_obs):
        return self.state_mlp(state_obs).unsqueeze(1)

    def _get_world_coordinates(self, depth_obs, pose_obs):
        depth = depth_obs.permute(0, 3, 1, 2) / 1000.0
        depth = F.interpolate(depth, size=(16, 16), mode="nearest-exact")
        
        world_coords, _ = get_3d_coordinates(depth, pose_obs, 64, 64, 64, 64)
        return world_coords.permute(0, 2, 3, 1)

    def forward(self, observations):
        # 1. Process individual inputs
        features = self._process_rgb(observations["rgb"])
        state = self._process_state(observations["state"])
        world_coords = self._get_world_coordinates(
            observations["depth"],
            observations["sensor_param"]["base_camera"]["extrinsic_cv"],
        )

        # 2. Reshape features and coordinates for Transformer
        world_coords_flat = world_coords.reshape(world_coords.shape[0], -1, 3)
        feats_flat = features.permute(0, 2, 3, 1).reshape(
            features.shape[0], -1, features.shape[1]
        )

        # 3. Pass through Transformer encoder and decoder
        visual_token = self.transformer_encoder(token=feats_flat, coords=world_coords_flat)
        action = self.transformer_decoder(memory=feats_flat, state=state).squeeze(1)

        return action


class Agent(nn.Module):
    """Actor-Critic network (Gaussian continuous actions) built on NatureCNN features."""

    def __init__(self, envs, sample_obs):
        super().__init__()
        self.feature_net = ActionModel(sample_obs= sample_obs)
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

    def get_features(self, x):
        return self.feature_net(x)

    def get_value(self, x):
        x = self.feature_net(x)
        return self.critic(x)

    def get_action(self, x, deterministic: bool = False):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x, action=None):
        x = self.feature_net(x)
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
