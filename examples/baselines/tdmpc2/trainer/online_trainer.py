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
		ep_rewards, ep_successes, ep_fails = [], [], []
		for i in range((self.cfg.eval_episodes - 1) // self.cfg.num_eval_envs + 1):
			obs, _ = self.eval_env.reset()
			done = torch.full((self.cfg.num_eval_envs, ), False, device=obs.device) # ms3: done is truncated since the ms3 ignore_terminations.
			ep_reward, t = torch.zeros((self.cfg.num_eval_envs, ), device=obs.device), 0
			if self.cfg.save_video:
				self.logger.video.init_cur_eps(self.eval_env, enabled=(i==0))
			while not done[0]: # done is truncated and should be the same
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, terminated, truncated, info = self.eval_env.step(action)
				done = terminated | truncated
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record_cur_eps(self.eval_env)
			ep_rewards.extend(ep_reward.tolist())

			if 'success' in info: 
				has_success = True
				ep_successes.extend(info['success'].float().tolist())
			
			if 'fail' in info:
				has_fail = True
				ep_fails.extend(info['fail'].float().tolist())

			if self.cfg.save_video:
				self.logger.video.save_cur_eps()
		if self.cfg.save_video:
			self.logger.video.flush_saved_eps(step=self._step, num_episodes=self.cfg.eval_episodes)
		# Truncate recorded episodes to self.cf.eval_episodes
		ep_rewards = ep_rewards[:self.cfg.eval_episodes]
		if has_success:
			ep_successes = ep_successes[:self.cfg.eval_episodes]
		if has_fail:
			ep_fails = ep_fails[:self.cfg.eval_episodes]

		# Update logger
		eval_metrics = dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_len=self.eval_env.max_episode_steps,
			episode_reward_avg=np.nanmean(ep_rewards)/self.eval_env.max_episode_steps,
		)
		if has_success:
			eval_metrics.update(episode_success=np.nanmean(ep_successes))
		if has_fail:
			eval_metrics.update(episode_fail=np.nanmean(ep_fails))
		return eval_metrics

	def to_td(self, obs, num_envs, action=None, reward=None):
		"""Before: Creates a TensorDict for a new episode. Return a td with batch (1, ), with obs, action, reward
		After vectorization: added 1 argument: num_envs, now have batch size (num_envs, 1)"""
		if isinstance(obs, dict): 
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
					eval_next = False

				if self._step > 0:
					tds = torch.cat(self._tds, dim=1) # [num_envs, episode_len + 1, ..]. Note it's different from official vectorized code
					train_metrics.update(
						episode_reward=tds['reward'].nansum(1).mean(), # first NaN is dropped by nansum
						episode_len=self.env.max_episode_steps,
						episode_reward_avg=tds['reward'].nansum(1).mean()/self.env.max_episode_steps,
					)
					if 'success' in vec_info:
						train_metrics.update(episode_success=vec_info['success'].float().mean().item())
					if 'fail' in vec_info:
						train_metrics.update(episode_fail=vec_info['fail'].float().mean().item())

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
			if vec_done[0] and self.cfg.obs == 'state': # added in vectorization
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
