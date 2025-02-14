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

	def __init__(self, cfg, env, num_envs, num_frames=3):
		self.base_env: BaseEnv = env.unwrapped
		super().__init__(env)
		self.cfg = cfg
		self.num_envs = num_envs
		self.num_frames = num_frames
		self.rgb_shape = self.base_env._init_raw_obs['rgb'].shape # (num_envs, h, w, c)
		self.rgb_stack = torch.zeros((*self.rgb_shape, self.num_frames), dtype=torch.uint8, device=self.env.device)
		if 'state' in self.base_env._init_raw_obs:
			self.state_shape = self.base_env._init_raw_obs['state'].shape # (num_envs, n)
			self.state_stack = torch.zeros((*self.state_shape, self.num_frames), dtype=torch.float32, device=self.env.device)
			self.include_state = True
		else:
			self.include_state = False
		self._stack_idx = 0
		self._render_size = cfg.render_size
		
		new_obs = self.observation(self.base_env._init_raw_obs)
		self.base_env.update_obs_space(new_obs)
		self.observation_space = self.base_env.observation_space
		self.single_observation_space = self.base_env.single_observation_space

	def observation(self, obs):
		self.rgb_stack[..., self._stack_idx] = obs['rgb']
		if self.include_state:
			self.state_stack[..., self._stack_idx] = obs['state']
		self._stack_idx = (self._stack_idx + 1) % self.num_frames
		rgb = self.rgb_stack.roll(shifts=-self._stack_idx, dims=-1).permute(0,1,2,4,3).reshape((*self.rgb_shape[:-1], -1)).permute(0, 3, 1, 2)		
		if self.include_state:
			state = self.state_stack.roll(shifts=-self._stack_idx, dims=-1).permute(0,2,1).reshape((*self.state_shape[:-1], -1))
			return {'rgb': rgb, 'rgb-state': state}
		else:
			return rgb

	def reset(self, *, seed=None, options=None):
		obs, info = self.env.reset()
		for _ in range(self.num_frames):
			obs_frames = self.observation(obs)
		return obs_frames, info

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action)
		return self.observation(obs), reward, terminated, truncated, info
