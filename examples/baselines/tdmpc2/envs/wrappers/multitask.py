import gym
import numpy as np
import torch


class MultitaskWrapper(gym.Wrapper):
	"""
	Wrapper for multi-task environments.
	"""

	def __init__(self, cfg, envs):
		super().__init__(envs[0])
		self.cfg = cfg
		self.envs = envs
		self._task = cfg.tasks[0]
		self._task_idx = 0
		self._obs_dims = [env.observation_space.shape[0] for env in self.envs]
		self._action_dims = [env.action_space.shape[0] for env in self.envs]
		self._episode_lengths = [env.max_episode_steps for env in self.envs]
		self._obs_shape = (max(self._obs_dims),)
		self._action_dim = max(self._action_dims)
		self.observation_space = gym.spaces.Box(
			low=-np.inf, high=np.inf, shape=self._obs_shape, dtype=np.float32
		)
		self.action_space = gym.spaces.Box(
			low=-1, high=1, shape=(self._action_dim,), dtype=np.float32
		)
	
	@property
	def task(self):
		return self._task
	
	@property
	def task_idx(self):
		return self._task_idx
	
	@property
	def _env(self):
		return self.envs[self.task_idx]

	def rand_act(self):
		return torch.from_numpy(self.action_space.sample().astype(np.float32))

	def _pad_obs(self, obs):
		if obs.shape != self._obs_shape:
			obs = torch.cat((obs, torch.zeros(self._obs_shape[0]-obs.shape[0], dtype=obs.dtype, device=obs.device)))
		return obs
	
	def reset(self, task_idx=-1):
		self._task_idx = task_idx
		self._task = self.cfg.tasks[task_idx]
		self.env = self._env
		return self._pad_obs(self.env.reset())

	def step(self, action):
		obs, reward, done, info = self.env.step(action[:self.env.action_space.shape[0]])
		return self._pad_obs(obs), reward, done, info
