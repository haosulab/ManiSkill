from collections import deque

import gymnasium as gym
import numpy as np
import torch


class PixelWrapper(gym.Wrapper):
	"""
	Wrapper for pixel observations. Works with Maniskill vectorized environments
	"""

	def __init__(self, cfg, env, num_envs, num_frames=3):
		super().__init__(env)
		self.cfg = cfg
		self.env = env
		self.observation_space = gym.spaces.Box(
			low=0, high=255, shape=(num_envs, num_frames*3, cfg.render_size, cfg.render_size), dtype=np.uint8
		)
		self._frames = deque([], maxlen=num_frames)
		self._render_size = cfg.render_size

		# # Using tensor to mimick self._frames = deque([], maxlen=num_frames) so the data remain on the same device **turned out to be slower**
		# self._frames = torch.zeros((num_envs, num_frames*3, cfg.render_size, cfg.render_size)).to(self.env.device)
		# self._frames_idx = 0
		# self._frames_maxlen = num_frames
		# self._render_size = cfg.render_size

	def _get_obs(self, obs):
		frame = obs['sensor_data']['base_camera']['rgb'].cpu().permute(0,3,1,2)
		self._frames.append(frame)
		return torch.from_numpy(np.concatenate(self._frames, axis=1)).to(self.env.device)
	
		# frame = obs['sensor_data']['base_camera']['rgb'].permute(0,3,1,2)
		# self._frames[:, self._frames_idx*3:self._frames_idx*3+3, ...] = frame
		# self._frames_idx = (self._frames_idx + 1) % self._frames_maxlen
		
		# return torch.cat((self._frames[:,self._frames_idx*3:,:], self._frames[:,:self._frames_idx*3,:]), 1) # reorder so obs is from old to new

	def reset(self):
		obs, info = self.env.reset()
		for _ in range(self._frames.maxlen):
			obs_frames = self._get_obs(obs)
		return obs_frames, info

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action)
		return self._get_obs(obs), reward, terminated, truncated, info
