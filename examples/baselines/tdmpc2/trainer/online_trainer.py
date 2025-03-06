from collections import defaultdict
from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict

from trainer.base import Trainer


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()

	def final_info_metrics(self, info):
		metrics = dict()
		if self.cfg.env_type == 'gpu':
			for k, v in info["final_info"]["episode"].items():
				metrics[k]=v.float().mean().item()
		else: # cpu
			temp = defaultdict(list)
			for final_info in info["final_info"]:
				for k, v in final_info["episode"].items():
					temp[k].append(v)
			for k, v in temp.items():
				metrics[k]=np.mean(v)
		return metrics

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		return dict(
			step=self._step,
			episode=self._ep_idx,
			total_time=time() - self._start_time,
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		has_success, has_fail = False, False # if task has success or/and fail (added for maniskill)
		for i in range(self.cfg.eval_episodes_per_env):
			obs, _ = self.eval_env.reset()
			done = torch.full((self.cfg.num_eval_envs, ), False, device=('cuda' if self.cfg.env_type=='gpu' else 'cpu')) # ms3: done is truncated since the ms3 ignore_terminations.
			ep_reward, t = torch.zeros((self.cfg.num_eval_envs, ), device=('cuda' if self.cfg.env_type=='gpu' else 'cpu')), 0
			while not done[0]: # done is truncated and should be the same
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, terminated, truncated, info = self.eval_env.step(action)
				done = terminated | truncated
				t += 1
		# Update logger
		eval_metrics = dict()
		eval_metrics.update(self.final_info_metrics(info))
		return eval_metrics

	def to_td(self, obs, num_envs, action=None, reward=None):
		"""Before: Creates a TensorDict for a new episode. Return a td with batch (1, ), with obs, action, reward
		After vectorization: added 1 argument: num_envs, now have batch size (num_envs, 1)"""
		if isinstance(obs, dict): 
			obs = {k: v.unsqueeze(1) for k,v in obs.items()}
			obs = TensorDict(obs, batch_size=(), device='cpu') # before vectorization, obs must have its first dimension=1
		else:
			obs = obs.unsqueeze(1).cpu()
		if action is None:
			action = torch.full((num_envs, self.cfg.action_dim), float('nan')) 
		if reward is None:
			reward = torch.full((num_envs,), float('nan'))
		td = TensorDict(dict(
			obs=obs,
			action=action.unsqueeze(1),
			reward=reward.cpu().unsqueeze(1),
		), batch_size=(num_envs, 1))
		return td

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, time_metrics, vec_done, eval_next = {}, {}, [True], True
		seed_finish = False

		rollout_times = []

		while self._step <= self.cfg.steps:

			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq < self.cfg.num_envs:
				eval_next = True

			# Reset environment
			if vec_done[0]:
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					self.logger.log_wandb_video(self._step)
					eval_next = False

				if self._step > 0:
					tds = torch.cat(self._tds, dim=1) # [num_envs, episode_len + 1, ..]. Note it's different from official vectorized code
					train_metrics.update(self.final_info_metrics(vec_info))
					if seed_finish:
						time_metrics.update(
							rollout_time=np.mean(rollout_times),
							rollout_fps=self.cfg.num_envs/np.mean(rollout_times), # self.cfg.num_envs * len(rollout_times)@steps_per_env@ /sum(rollout_times)
							update_time=update_time,
						)
						time_metrics.update(self.common_metrics())
						self.logger.log(time_metrics, 'time')
						rollout_times = []

					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')

					assert len(self._tds) == self.env.max_episode_steps + 1, f"{len(self._tds)} instead" # ManiSkillVectorEnv wrapper required
					self._ep_idx = self.buffer.add(tds) 

				obs, _ = self.env.reset()
				self._tds = [self.to_td(obs, self.cfg.num_envs)]

			# Collect experience
			rollout_time = time()
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds)==1, eval_mode=False) # t0 unchanged since all envs have same episode length
			else:
				# action = torch.rand((self.cfg.num_envs, self.cfg.action_dim)) # self.env.rand_act()
				action = torch.from_numpy(self.env.action_space.sample())
			obs, reward, vec_terminated, vec_truncated, vec_info = self.env.step(action)

			vec_done = vec_terminated | vec_truncated

			if vec_done[0]: # use actual final_observation
				if self.cfg.obs == 'rgb' and isinstance(vec_info["final_observation"], dict):
					obs = vec_info["final_observation"].copy()
				else:
					obs = vec_info["final_observation"]
			
			self._tds.append(self.to_td(obs, self.cfg.num_envs, action, reward))
			rollout_time = time() - rollout_time
			rollout_times.append(rollout_time)
			
			# Update agent
			if self._step >= self.cfg.seed_steps:
				update_time = time()
				if not seed_finish:
					seed_finish = True
					num_updates = int(self.cfg.seed_steps / self.cfg.steps_per_update)
					print('Pretraining agent on seed data...')
				else:
					num_updates = max(1, int(self.cfg.num_envs / self.cfg.steps_per_update))
				for _ in range(num_updates):
					_train_metrics = self.agent.update(self.buffer)
				train_metrics.update(_train_metrics)
				update_time = time() - update_time

			self._step += self.cfg.num_envs
	
		self.logger.finish(self.agent)
