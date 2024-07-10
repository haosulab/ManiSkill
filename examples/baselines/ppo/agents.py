import gymnasium as gym
import numpy as np

from typing import Dict

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from nets import *

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common





class FlattenPointcloudObservationWrapper(gym.ObservationWrapper):
    """
    Flattens the pointcloud mode observations into a dictionary with two keys, "pointcloud" and "state"
    """

    def __init__(self, env, pointcloud_only=False) -> None:
        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)

        self.pointcloud_only = pointcloud_only
        self.SAVE_TRANSFORM = False

        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: Dict):
        sensor_data = observation.pop("pointcloud")

        if self.SAVE_TRANSFORM:
            print("Saving cam2world from pcd task")
            assert "base_camera" in observation["sensor_param"], "there is a base camera sensor defined"
            cam2world = observation["sensor_param"]["base_camera"]["cam2world_gl"][0].cpu().numpy()
            np.save("cam2world_TEST.npy", cam2world)

        del observation["sensor_param"]

        points3d = []
        for type, data in sensor_data.items():
            if type =="xyzw":
                xyz = data[..., :3]
                points3d.append(xyz)

            if not self.pointcloud_only and type == "rgb":
                last_xyz = points3d[-1]
                pnt_rgb = data
                xyz_with_rgb = torch.concat([last_xyz, pnt_rgb], axis=-1)
                points3d[-1] = xyz_with_rgb

        points3d = torch.concat(points3d, axis=0)

        # flatten the rest of the data which should just be state data
        observation = common.flatten_state_dict(observation, use_torch=True)
        return dict(state=observation, pcd=points3d)
        
class FlattenDepthObservationWrapper(gym.ObservationWrapper):
    """
    Flattens the rgbd mode observations into a dictionary with two keys, "rgbd" and "state"
    """

    def __init__(self, env, with_rgb=False) -> None:
        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)

        self.with_rgb = with_rgb
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: Dict):
        sensor_data = observation.pop("sensor_data")
        del observation["sensor_param"]
        images = []
        for cam_data in sensor_data.values():
            images.append(cam_data["depth"])
            if self.with_rgb:
                images.append(cam_data["rgb"])

        images = torch.concat(images, axis=-1)
        
        # flatten the rest of the data which should just be state data
        observation = common.flatten_state_dict(observation, use_torch=True)
        return dict(state=observation, depth=images)
        

def compute_GAE(t, next_not_done, gae_lambda, gamma, rewards, real_next_values, values, lam_coef_sum, reward_term_sum, value_term_sum):
    """
    See GAE paper equation(16) line 1, we will compute the GAE based on this line only
    1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
    lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
    lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
    lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
    We then normalize it by the sum of the lambda^i (instead of 1-lambda)
    """
    
    lam_coef_sum = lam_coef_sum * next_not_done
    reward_term_sum = reward_term_sum * next_not_done
    value_term_sum = value_term_sum * next_not_done

    lam_coef_sum = 1 + gae_lambda * lam_coef_sum
    reward_term_sum = gae_lambda * gamma * reward_term_sum + lam_coef_sum * rewards[t]
    value_term_sum = gae_lambda * gamma * value_term_sum + gamma * real_next_values

    advantage = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
    return advantage, lam_coef_sum, reward_term_sum, value_term_sum


class StateAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()

        self.critic = StateCritic(envs.single_observation_space)

        # self.critic = nn.Sequential(
        #     layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(256, 256)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(256, 256)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(256, 1)),
        # )

        self.actor_mean = StateActor(envs.single_observation_space, envs.single_action_space)

        # self.actor_mean = nn.Sequential(
        #     layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(256, 256)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(256, 256)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(256, np.prod(envs.single_action_space.shape)), std=0.01*np.sqrt(2)),
        # )

        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(envs.single_action_space.shape)) * -0.5)

    def get_value(self, x):
        return self.critic(x)
    
    def get_action(self, x, deterministic=False):
        action_mean = self.actor_mean(x)
        
        if deterministic:
            return action_mean
        
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
    
    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


class Agent(nn.Module):
    def __init__(self, envs, sample_obs, feature_net=None, is_tracked=False, with_state=False, pretrained=False):
        super().__init__()

        print(f"Running with state: {with_state}")

        self.is_tracked = is_tracked

        #self.feature_net = NatureCNN(sample_obs=sample_obs, with_state=with_state, pretrained=pretrained)
        print("\t\tUsing NatureCNNGRURGBD\n\n\n")
        #self.feature_net = NatureCNNGRU(sample_obs=sample_obs, with_state=with_state, pretrained=pretrained)
        self.feature_net = NatureCNNGRURGBD(sample_obs=sample_obs, with_state=with_state, pretrained=pretrained)
        if feature_net:
            feature_net = feature_net
        
        # latent_size = np.array(envs.unwrapped.single_observation_space.shape).prod()
        latent_size = self.feature_net.out_features
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        
        STD_SCALE = 0.01
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            #nn.Dropout1d(p=0.3), # experiment
            layer_init(nn.Linear(512, np.prod(envs.unwrapped.single_action_space.shape)), std=STD_SCALE * np.sqrt(2)),
        )
        
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(envs.unwrapped.single_action_space.shape)) * -0.5)
    
    #def get_features(self, x):
    #    return self.feature_net(x)
    
    def get_value(self, x):
        x = self.feature_net(x)
        return self.critic(x)
    
    def get_action(self, x, deterministic=False):
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
        
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    
class PointcloudAgent:
    def __init__(self, envs, sample_obs, is_tracked=False, with_rgb=False):
        self.feature_encoder = PcdEncoder(sample_obs=sample_obs, normal_channel=False) if not with_rgb else \
            PcdEncoder(sample_obs=sample_obs, normal_channel=False) # TODO: Change to accomodate rgb codes for each pcd
        self.agent = Agent(envs=envs, sample_obs=sample_obs, feature_net=self.feature_encoder, is_tracked=is_tracked)

class RGBDAgent:
    def __init__(self, envs, sample_obs, is_tracked=False, with_rgb=False, with_state=False):
        self.feature_encoder = NatureCNN3D(sample_obs=sample_obs, with_rgb=with_rgb, with_state=with_state)
        print(f"Running with state: {with_state}")
        self.agent = Agent(envs=envs, sample_obs=sample_obs, feature_net=self.feature_encoder, is_tracked=is_tracked)
