import gymnasium as gym
import numpy as np
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils import gym_utils

import mani_skill.envs

def make_envs(cfg, num_envs):
	"""
	Make ManiSkill3 environment.
	"""
	#assert cfg.obs == 'state', 'This task only supports state observations.'
	if cfg.control_mode == 'default':
		env = gym.make(
			cfg.env_id,
			obs_mode=cfg.obs, 
			render_mode=cfg.render_mode,
			sensor_configs=dict(width=cfg.render_size, height=cfg.render_size),
			num_envs=num_envs
		)
	else:
		env = gym.make(
			cfg.env_id,
			obs_mode=cfg.obs,
			control_mode=cfg.control_mode,
			render_mode=cfg.render_mode,
			sensor_configs=dict(width=cfg.render_size, height=cfg.render_size),
			num_envs=num_envs
		)
	cfg.env_cfg.control_mode = cfg.eval_env_cfg.control_mode = env.control_mode
	env = ManiSkillVectorEnv(env, ignore_terminations=True)
	cfg.env_cfg.env_horizon = cfg.eval_env_cfg.env_horizon = env.max_episode_steps = gym_utils.find_max_episode_steps_value(env)
	return env