from copy import deepcopy
import warnings

import gymnasium as gym

from envs.wrappers.pixels import PixelWrapper

def missing_dependencies(task):
	raise ValueError(f'Missing dependencies for task {task}; install dependencies to use this environment.')

try:
	from envs.maniskill import make_envs as make_maniskill_vec_env
except:
	make_maniskill_env = missing_dependencies


warnings.filterwarnings('ignore', category=DeprecationWarning)

def make_envs(cfg):
	from envs.maniskill import make_envs as make_maniskill_vec_env
	env = make_maniskill_vec_env(cfg)

	if cfg.get('obs', 'state') == 'rgb':
		env = PixelWrapper(cfg, env)

	try: # Dict
		cfg.obs_shape = {k: v.shape[1:] for k, v in env.observation_space.spaces.items()}
	except: # Box
		cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape[1:]}
	cfg.action_dim = env.action_space.shape[1]
	cfg.episode_length = env.max_episode_steps
	cfg.seed_steps = max(1000, cfg.num_envs * cfg.episode_length)
	return env