import numpy as np

import torch
import torch.nn as nn

from torchvision.models import resnet18, resnet50
import torchvision.transforms as transforms
from torchvision import transforms

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class StateCritic(nn.Module):
    def __init__(self, observation_space):
        super(StateCritic, self).__init__()

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )

    def forward(self, x):
        return self.critic(x)
    
class StateCriticV1(nn.Module):
    def __init__(self, observation_space):
        super(StateCriticV1, self).__init__()

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_space.shape).prod(), 256)),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1)),
        )

    def forward(self, x):
        return self.critic(x)
    
class StateActor(nn.Module):
    def __init__(self, observation_space, action_space):
        super(StateActor, self).__init__()

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(action_space.shape)), std=0.01*np.sqrt(2)),
        )

    def forward(self, x):
        return self.actor_mean(x)
    
class StateActorV1(nn.Module):
    def __init__(self, observation_space, action_space):
        super(StateActorV1, self).__init__()

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_space.shape).prod(), 256)),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            layer_init(nn.Linear(256, np.prod(action_space.shape)), std=0.01*np.sqrt(2)),
        )

    def forward(self, x):
        return self.actor_mean(x)

class NatureCNN(nn.Module):
    def __init__(self, sample_obs, with_state=False, pretrained=False):
        super().__init__()

        self.pretrained = pretrained

        # NOTE: Run rgb or rgb + state
        self.with_state = with_state

        extractors = {}

        self.out_features = 0
        feature_size = 256
        in_channels = sample_obs["rgb"].shape[-1]
        image_size = (sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])
        state_size = sample_obs["state"].shape[-1]

        # here we use a NatureCNN architecture to process images, but any architecture is permissble here
        cnn = None
        if not self.pretrained:
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
        else:
            print("Using pretrained resnet18")

            resnet = resnet18(pretrained=False) # False or True, whatever works
            # Remove the last fully connected layer
            cnn = nn.Sequential(
                *list(resnet.children())[:-1], 
                nn.Flatten()
            )

            #self.preprocess = transforms.Compose([
            #     transforms.Resize(224),
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # ])

        # to easily figure out the dimensions after flattening, we pass a test tensor
        with torch.no_grad():
            n_flatten = cnn(sample_obs["rgb"].float().permute(0,3,1,2).cpu()).shape[1]
            if pretrained:
                sample_input = sample_obs["rgb"].resize_(
                    sample_obs["rgb"].shape[0], 
                    224, 
                    224, 
                    sample_obs["rgb"].shape[-1]
                )
                sample_input = sample_input.float().permute(0,3,1,2).cpu()
                sample_out = cnn(sample_input)
                n_flatten = sample_out.shape[1]
            fc = nn.Sequential(
                nn.Linear(n_flatten, feature_size), 
                nn.ReLU()
            )

        extractors["rgb"] = nn.Sequential(cnn, fc)
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
            if key == "rgb":
                obs = obs.float().permute(0,3,1,2)
                #original_h, original_w = obs.shape[0], obs.shape[1]
                if not self.pretrained:
                    obs = obs / 255
                else:
                    obs = obs.resize_(obs.shape[0], obs.shape[1], 224, 224)
                    #mean = [0.485, 0.456, 0.406]
                    #std = [0.229, 0.224, 0.225]
                    #normalize_fn = transforms.Normalize(mean=mean, std=std)
                    #obs = normalize_fn(obs) (only when pretrained=True)
                    obs = obs / 255
                    
                    #obs[:, 0, ...] = (obs[:, 0, ...] - mean[0]) / std[0]
                    # obs[:, 1, ...] = (obs[:, 1, ...] - mean[1]) / std[1]
                    # obs[:, 2, ...] = (obs[:, 2, ...] - mean[2]) / std[2]
            out = extractor(obs)
            encoded_tensor_list.append(out)
        return torch.cat(encoded_tensor_list, dim=1)

class NatureCNNGRU(nn.Module):
    def __init__(self, sample_obs, hidden_dim=256, with_state=False, pretrained=None):
        super().__init__()

        # NOTE: Run rgb or rgb + state
        self.with_state = with_state

        self.out_features = 0
        feature_size = 256
        in_channels = sample_obs["rgb"].shape[-1]
        image_size = (sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])
        state_size = sample_obs["state"].shape[-1]

        # here we use a NatureCNN architecture to process images, but any architecture is permissble here
        self.cnn = nn.Sequential(
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
            #nn.Dropout2d(p=0.3, inplace=True), # experiment
            nn.Flatten()
        )

        with torch.no_grad():
            n_size = self.cnn(sample_obs["rgb"].float().permute(0,3,1,2).cpu())
            n_size = n_size.shape[1]
            self.gru = nn.Sequential(
                nn.GRU(n_size, hidden_dim, batch_first=True),
            )

        # to easily figure out the dimensions after flattening, we pass a test tensor
        with torch.no_grad():
            n_flatten = self.cnn(sample_obs["rgb"].float().permute(0,3,1,2).cpu())
            n_flatten, _ = self.gru(n_flatten)
            n_flatten = n_flatten.shape[1]

            self.fc = nn.Sequential(
                nn.Linear(n_flatten, feature_size), 
                nn.ReLU()
            )

        self.out_features += feature_size

        # for state data we simply pass it through a single linear layer
        self.fc_state = None
        if self.with_state:
            self.fc_state = nn.Linear(state_size, 256)
            self.out_features += 256

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        obs = observations["rgb"]
        obs = obs.float().permute(0,3,1,2)
        obs = obs / 255
        out = self.cnn(obs)
        out, _ = self.gru(out)
        out = self.fc(out)
        encoded_tensor_list.append(out)

        if self.with_state:
            out = self.fc_state(out)
            encoded_tensor_list.append(out)

        return torch.cat(encoded_tensor_list, dim=1)

class NatureCNNGRURGBD(nn.Module):
    def __init__(self, sample_obs, hidden_dim=256, with_state=False, pretrained=None):
        super().__init__()

        # NOTE: Run rgb or rgb + state
        self.with_state = with_state

        self.out_features = 0
        feature_size = 256
        in_channels = sample_obs["rgbd"].shape[-1]
        image_size = (sample_obs["rgbd"].shape[1], sample_obs["rgbd"].shape[2])
        state_size = sample_obs["state"].shape[-1]

        # here we use a NatureCNN architecture to process images, but any architecture is permissble here
        self.cnn = nn.Sequential(
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
                in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),


            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            n_size = self.cnn(sample_obs["rgbd"].float().permute(0,3,1,2).cpu())
            n_size = n_size.shape[1]
            self.gru = nn.Sequential(
                nn.GRU(n_size, hidden_dim, batch_first=True),
            )

        # to easily figure out the dimensions after flattening, we pass a test tensor
        with torch.no_grad():
            n_flatten = self.cnn(sample_obs["rgbd"].float().permute(0,3,1,2).cpu())
            n_flatten, _ = self.gru(n_flatten)
            n_flatten = n_flatten.shape[1]
            self.fc = nn.Sequential(
                nn.Linear(n_flatten, feature_size), 
                nn.ReLU()
            )

        self.out_features += feature_size

        # for state data we simply pass it through a single linear layer
        self.fc_state = None
        if self.with_state:
            self.fc_state = nn.Linear(state_size, 256)
            self.out_features += 256

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # RGB
        obs_rgb = observations["rgbd"][..., 1:]
        obs_rgb = obs_rgb.float().permute(0,3,1,2)
        obs_rgb = obs_rgb / 255

        # Depth
        obs_depth = observations["rgbd"][..., :1]
        obs_depth = obs_depth.float().permute(0,3,1,2)
        obs_depth = obs_depth / 65535

        obs = torch.concat([obs_rgb, obs_depth], axis=1)

        out = self.cnn(obs)
        out, _ = self.gru(out)
        out = self.fc(out)
        encoded_tensor_list.append(out)

        if self.with_state:
            out = self.fc_state(out)
            encoded_tensor_list.append(out)

        return torch.cat(encoded_tensor_list, dim=1)


#TODO
class NatureCNN3D(nn.Module):
    """ Supports rgbd """
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
                depth = sample_obs["depth"][..., :1].float().permute(0,3,1,2).cpu()
                rgb = sample_obs["depth"][..., 1:].float().permute(0,3,1,2).cpu()
                input_data = torch.concat([depth, rgb], axis=1)

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
                    range = (depth.min(), depth.max()) # debug
                    obs = torch.concat([depth, rgb], axis=1)
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