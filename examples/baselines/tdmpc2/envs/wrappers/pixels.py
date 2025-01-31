from collections import deque

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces.dict import Dict
from gymnasium.spaces import Box
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common

class PixelWrapper(gym.ObservationWrapper):
	"""
	Wrapper for pixel observations. Works with Maniskill vectorized environments
	"""

	def __init__(self, cfg, env, num_envs, state=True, num_frames=3):
		self.base_env: BaseEnv = env.unwrapped
		super().__init__(env)
		self.cfg = cfg
		self.rgb_frames = deque([], maxlen=num_frames)
		self.state_frames = deque([], maxlen=num_frames)
		self._render_size = cfg.render_size
		self.include_state = state

		for _ in range(self.rgb_frames.maxlen):
			new_obs = self.observation(self.base_env._init_raw_obs)
		self.base_env.update_obs_space(new_obs)
		self.observation_space = self.base_env.observation_space
		self.single_observation_space = self.base_env.single_observation_space
		# self.observation_space = gym.spaces.Box(
		# 	low=0, high=255, shape=(num_envs, num_frames*3, cfg.render_size, cfg.render_size), dtype=np.uint8
		# )

	def observation(self, obs: Dict):
		rgb_frame = obs["sensor_data"]['base_camera']['rgb'].cpu().permute(0,3,1,2)
		filtered_obs = {k: v for k, v in obs.items() if k not in ["sensor_data", "sensor_param"]}
		state_frame = common.flatten_state_dict(
			filtered_obs, use_torch=True, device=self.base_env.device
		)
		self.state_frames.append(state_frame.cpu())
		self.rgb_frames.append(rgb_frame)
		ret = dict()
		ret["rgb"] = torch.from_numpy(np.concatenate(self.rgb_frames, axis=1)).to(self.base_env.device)
		if self.include_state:
			ret["state"] = torch.from_numpy(np.concatenate(self.state_frames, axis=1)).to(self.base_env.device)
		return ret

	def reset(self, *, seed=None, options=None):
		obs, info = self.env.reset()
		for _ in range(self.rgb_frames.maxlen):
			obs_frames = self.observation(obs)
		return obs_frames, info

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action)
		return self.observation(obs), reward, terminated, truncated, info
