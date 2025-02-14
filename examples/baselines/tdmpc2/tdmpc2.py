import numpy as np
import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel


class TDMPC2:
	"""
	TD-MPC2 agent. Implements training + inference.
	Can be used for both single-task and multi-task experiments,
	and supports both state and pixel observations.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self.device = torch.device('cuda')
		self.model = WorldModel(cfg).to(self.device)
		self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._Qs.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []}
		], lr=self.cfg.lr)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda'
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		
		self.cfg.discount = self.discount

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.
		
		Args:
			fp (str): Filepath to save state dict to.
		"""
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.
		
		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		state_dict = fp if isinstance(fp, dict) else torch.load(fp)
		self.model.load_state_dict(state_dict["model"])

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Before: obs is 1d, return seems to be mu(1, action_dim)
		After: obs is batched with num_env, return still 2d

		Select an action by planning in the latent space of the world model.
		
		Args:
			obs (torch.Tensor): Observation from the environment. 1d for online trainer
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (int): Task index (only used for multi-task experiments).
		
		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		if isinstance(obs, dict): # RGB
			obs = {k: v.to(self.device, non_blocking=True) for k,v in obs.items()}
		else:
			obs = obs.to(self.device, non_blocking=True)
		if task is not None:
			task = torch.tensor([task], device=self.device)
		z = self.model.encode(obs, task) # [num_envs, latent_dim]
		if self.cfg.mpc:
			a = self.plan(z, t0=t0, eval_mode=eval_mode, task=task)
		else:
			a = self.model.pi(z, task)[int(not eval_mode)] # [int(not eval_mode)] selects mu or pi
		return a.cpu()

	@torch.no_grad()
	def _estimate_value(self, z, actions, task):
		"""z[num_samples, latent_dim], actions[horizon, num_samples, action_dim] -> [num_samples, 1]
		Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[:, t], task), self.cfg)
			z = self.model.next(z, actions[:, t], task)
			G += discount * reward
			discount *= self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
		return G + discount * self.model.Q(z, self.model.pi(z, task)[1], task, return_type='avg')

	@torch.no_grad()
	def plan(self, z, t0=False, eval_mode=False, task=None):
		"""
		Before: For online, z[1, latent_dim]
		After: For online z[num_envs, latent_dim]. Should be ok
		Plan a sequence of actions using the learned world model.
		
		Args:
			z (torch.Tensor): Latent state from which to plan.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""	
		num_envs = self.cfg.num_eval_envs if eval_mode else self.cfg.num_envs
		# Sample policy trajectories
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(num_envs, self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.unsqueeze(1).repeat(1, self.cfg.num_pi_trajs, 1) # (num_envs, num_pi_trajs, latent_dim)
			for t in range(self.cfg.horizon-1):
				pi_actions[:, t] = self.model.pi(_z, task)[1]
				_z = self.model.next(_z, pi_actions[:, t], task)
			pi_actions[:, -1] = self.model.pi(_z, task)[1]

		# Initialize state and parameters
		z = z.unsqueeze(1).repeat(1, self.cfg.num_samples, 1) # (num_envs, num_samples, latent_dim)
		mean = torch.zeros(num_envs, self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = self.cfg.max_std*torch.ones(num_envs, self.cfg.horizon, self.cfg.action_dim, device=self.device)
		if not t0 and hasattr(self, '_prev_mean'):
			if eval_mode: # Added to avoid the problem with shape (num_envs) mismatch with train and eval env
				mean[:, :-1] = self._prev_mean_eval[:, 1:]
			else:
				mean[:, :-1] = self._prev_mean[:, 1:]
		actions = torch.empty(num_envs, self.cfg.horizon, self.cfg.num_samples, 
						self.cfg.action_dim, device=self.device) # # (num_envs, horizon, num_samples, latent_dim)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :, :self.cfg.num_pi_trajs] = pi_actions
	
		# Iterate MPPI
		for _ in range(self.cfg.iterations):

			# Sample actions
			actions[:, :, self.cfg.num_pi_trajs:] = (mean.unsqueeze(2) + std.unsqueeze(2) * \
				torch.randn(num_envs, self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)) \
				.clamp(-1, 1)
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			# Compute elite actions
			value = self._estimate_value(z, actions, task).nan_to_num_(0) # (num_envs, num_samples, 1)
			elite_idxs = torch.topk(value.squeeze(2), self.cfg.num_elites, dim=1).indices # (num_envs, num_elites)
			elite_value = value[torch.arange(num_envs).unsqueeze(1), elite_idxs] # (num_envs, num_elites, 1)
			# elite_actions = torch.zeros(num_envs, self.cfg.horizon, self.cfg.num_elites, self.cfg.action_dim, dtype=actions.dtype, device=actions.device)
			# for j, curr_elites in enumerate(elite_idxs):
			# 	elite_actions[j] = actions[j, :, curr_elites]
			elite_actions = torch.gather(actions, 2, elite_idxs.unsqueeze(1).unsqueeze(3).expand(-1, self.cfg.horizon, -1, self.cfg.action_dim))

			# Update parameters
			max_value = elite_value.max(1)[0] # (num_envs, 1)
			score = torch.exp(self.cfg.temperature*(elite_value - max_value.unsqueeze(1)))
			score /= score.sum(1, keepdim=True) # (num_envs, num_elites, 1)
			mean = torch.sum(score.unsqueeze(1) * elite_actions, dim=2) / (score.sum(1, keepdim=True) + 1e-9)  # (num_envs, horizon, action_dim)
			std = torch.sqrt(torch.sum(score.unsqueeze(1) * (elite_actions - mean.unsqueeze(2)) ** 2, dim=2) / (score.sum(1, keepdim=True) + 1e-9)) \
				.clamp_(self.cfg.min_std, self.cfg.max_std) # (num_envs, horizon, action_dim)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]

		# Select action
		score = score.squeeze(2).cpu().numpy() # (num_envs, num_elites)
		# (num_envs, horizon, num_elites, action_dim) for elite_actions
		actions = torch.zeros(num_envs, self.cfg.horizon, self.cfg.action_dim, dtype=actions.dtype, device=actions.device)
		for i in range(len(score)):
			actions[i] = elite_actions[i, :, np.random.choice(np.arange(score.shape[1]), p=score[i])]
		if eval_mode:
			self._prev_mean_eval = mean # (num_eval_envs, horizon, action_dim)
		else:
			self._prev_mean = mean # (num_envs, horizon, action_dim)
		a, std = actions[:, 0], std[:, 0]
		if not eval_mode:
			a += std * torch.randn(num_envs, self.cfg.action_dim, device=std.device)
		return a.clamp_(-1, 1)
		
	def update_pi(self, zs, task):
		"""
		Update policy using a sequence of latent states.
		
		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		self.pi_optim.zero_grad(set_to_none=True)
		self.model.track_q_grad(False)
		_, pis, log_pis, _ = self.model.pi(zs, task)
		qs = self.model.Q(zs, pis, task, return_type='avg')
		self.scale.update(qs[0])
		qs = self.scale(qs)

		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.model.track_q_grad(True)

		return pi_loss.item()

	@torch.no_grad()
	def _td_target(self, next_z, reward, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.
		
		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).
		
		Returns:
			torch.Tensor: TD-target.
		"""
		pi = self.model.pi(next_z, task)[1]
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		return reward + discount * self.model.Q(next_z, pi, task, return_type='min', target=True)

	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.
		
		Args:
			buffer (common.buffer.Buffer): Replay buffer.
		
		Returns:
			dict: Dictionary of training statistics.
		"""
		obs, action, reward, task = buffer.sample()
	
		# Compute targets
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			td_targets = self._td_target(next_z, reward, task)

		# Prepare for update
		self.optim.zero_grad(set_to_none=True)
		self.model.train()

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.true_latent_dim, device=self.device)
		z = self.model.encode(obs[0], task)
		zs[0] = z
		consistency_loss = 0
		for t in range(self.cfg.horizon):
			z = self.model.next(z, action[t], task)
			consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t
			zs[t+1] = z

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, task, return_type='all')
		reward_preds = self.model.reward(_zs, action, task)
		
		# Compute losses
		reward_loss, value_loss = 0, 0
		for t in range(self.cfg.horizon):
			reward_loss += math.soft_ce(reward_preds[t], reward[t], self.cfg).mean() * self.cfg.rho**t
			for q in range(self.cfg.num_q):
				value_loss += math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean() * self.cfg.rho**t
		consistency_loss *= (1/self.cfg.horizon)
		reward_loss *= (1/self.cfg.horizon)
		value_loss *= (1/(self.cfg.horizon * self.cfg.num_q))
		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.value_coef * value_loss
		)

		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()

		# Update policy
		pi_loss = self.update_pi(zs.detach(), task)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		return {
			"consistency_loss": float(consistency_loss.mean().item()),
			"reward_loss": float(reward_loss.mean().item()),
			"value_loss": float(value_loss.mean().item()),
			"pi_loss": pi_loss,
			"total_loss": float(total_loss.mean().item()),
			"grad_norm": float(grad_norm),
			"pi_scale": float(self.scale.value),
		}
