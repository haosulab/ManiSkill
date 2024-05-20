import numpy as np

import torch
import torch.nn as nn

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class NatureCNN(nn.Module):
    def __init__(self, sample_obs, is_rgb_only=False):
        super().__init__()

        # NOTE: Run rgb or rgb + state
        self.is_rgb_only = is_rgb_only

        extractors = {}

        self.out_features = 0
        feature_size = 256
        in_channels = sample_obs["rgb"].shape[-1]
        image_size = (sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])
        state_size = sample_obs["state"].shape[-1]

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

        # for state data we simply pass it through a single linear layer
        if not self.is_rgb_only:
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                obs = obs.float().permute(0,3,1,2)
                obs = obs / 255
            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)

#TODO
class NatureCNN3D(nn.Module):
    """ Supports rgbb """
    def __init__(self, sample_obs, with_rgb=False, with_state=False):
        super().__init__()

        # NOTE: Run rgb or rgb + state
        self.with_rgb = with_rgb
        self.with_state = with_state

        extractors = {}

        self.out_features = 0
        feature_size = 256
        in_channels = sample_obs["depth"].shape[-1]
        state_size = sample_obs["state"].shape[-1]

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
            input_data = sample_obs["depth"].float().permute(0,3,1,2).cpu()
            if self.with_rgb:
                depth = sample_obs["depth"].float().permute(0,3,1,2).cpu()
                rgb = sample_obs["rgb"].float().permute(0,3,1,2).cpu()
                input_data = torch.concat([depth, rgb], axis=-1)

            n_flatten = cnn(input_data).shape[1]
            fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())

        extractors["depth"] = nn.Sequential(cnn, fc)
        self.out_features += feature_size

        # for state data we simply pass it through a single linear layer
        if self.with_state:
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "depth":
                if self.with_rgb:
                    rgb = obs[..., 1:].float().permute(0,3,1,2) / 255
                    depth = obs[..., :1].float().permute(0,3,1,2) / 65535
                    obs = torch.concat([depth, rgb], axis=-1)
                else:
                    # TODO: Check the range of the depth
                    obs = obs[..., :1].float().permute(0,3,1,2) / 65535

            elif key == "state" and not self.with_state:
                continue

            encoded_tensor_list.append(extractor(obs))
        
        return torch.cat(encoded_tensor_list, dim=1) 

# Pointcloud
from net_utils import *

class PcdEncoder(nn.Module):
    def __init__(self, sample_obs, normal_channel=False):
        super(PcdEncoder, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        
        self.extractors = {}
        self.out_features = 256

        state_size = sample_obs["state"].shape[-1]

        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, self.out_features)
        self.bn2 = nn.BatchNorm1d(self.out_features)
        self.drop2 = nn.Dropout(0.4)

    def forward(self, x):
        xyz = x["pcd"].permute(0, 2, 1) #[B, N, C] -> [B, C, N] for the PointNet++; consistency
        state = x["state"]
        B, N, C = xyz.shape

        #TODO: Adapt for any sort of additional conditioning
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        x = l3_points.view(B, 1024)
        
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        

        return x#, l3_points