import gymnasium as gym
import numpy as np
from common.logger import Logger
from envs.wrappers.pixels import PixelWrapper
from envs.wrappers.tensor import TensorWrapper
from envs.wrappers.record_episode import RecordEpisodeWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
from mani_skill.utils.wrappers import FlattenRGBDObservationWrapper
from mani_skill.utils import gym_utils
from functools import partial
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv, VectorEnv

import mani_skill.envs

def cpu_env_factory(env_make_fn, idx: int, wrappers=[], record_video_path: str = None, record_episode_kwargs=dict(), logger: Logger = None):
	def _init():
		env = env_make_fn()
		for wrapper in wrappers:
			env = wrapper(env)
		env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
		if record_video_path is not None and (not record_episode_kwargs["record_single"] or idx == 0):
			env = RecordEpisodeWrapper(
                env,
                record_video_path,
                trajectory_name=f"trajectory_{idx}",
                save_video=record_episode_kwargs["save_video"],
                save_trajectory=record_episode_kwargs["save_trajectory"],
                info_on_video=record_episode_kwargs["info_on_video"],
                logger=logger,
            )
		return env

	return _init

def make_envs(cfg, num_envs, record_video_path, is_eval, logger):
	"""
	Make ManiSkill3 environment.
	"""
	record_episode_kwargs = dict(save_video=True, save_trajectory=False, record_single=True, info_on_video=False)

	# Set up env make fn for consistency
	env_make_fn = partial(
		gym.make, 
		disable_env_checker=True,
		id=cfg.env_id, 
		obs_mode=cfg.obs, 
		render_mode=cfg.render_mode, 
		sensor_configs=dict(width=cfg.render_size, height=cfg.render_size)
		)
	if cfg.control_mode != 'default':
		env_make_fn = partial(env_make_fn, control_mode=cfg.control_mode)
	if is_eval: # https://maniskill.readthedocs.io/en/latest/user_guide/reinforcement_learning/setup.html#evaluation
		env_make_fn = partial(env_make_fn, reconfiguration_freq=cfg.eval_reconfiguration_frequency)

	if cfg.env_type == 'cpu':
		# Get default control_mode and max_episode_steps values
		dummy_env = env_make_fn()
		control_mode = dummy_env.control_mode
		max_episode_steps = gym_utils.find_max_episode_steps_value(dummy_env)
		dummy_env.close()
		del dummy_env
		# Create cpu async vectorized env
		vector_env_cls = partial(AsyncVectorEnv, context="forkserver")
		if num_envs == 1:
			vector_env_cls = SyncVectorEnv
		wrappers = []
		if cfg['obs'] == 'rgb':
			wrappers.append(partial(PixelWrapper(cfg=cfg, num_envs=num_envs)))
		env: VectorEnv = vector_env_cls(
			[
				cpu_env_factory(env_make_fn, i, wrappers, record_video_path, record_episode_kwargs, logger)
				for i in range(num_envs)
			]
		)
		env = TensorWrapper(env)
	elif cfg.env_type == 'gpu':
		env = env_make_fn(num_envs=num_envs)
		control_mode = env.control_mode
		max_episode_steps = gym_utils.find_max_episode_steps_value(env)
		if cfg['obs'] == 'rgb':
			env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=cfg.include_state)
			env = PixelWrapper(cfg, env, num_envs)
		if record_video_path is not None:
			env = RecordEpisodeWrapper(
					env,
					record_video_path,
					trajectory_name=f"trajectory",
					max_steps_per_video=max_episode_steps,
					save_video=record_episode_kwargs["save_video"],
					save_trajectory=record_episode_kwargs["save_trajectory"],
					logger=logger,
				)
		env = ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)
	else:
		raise Exception('env_type must be cpu or gpu')
	cfg.env_cfg.control_mode = cfg.eval_env_cfg.control_mode = control_mode
	cfg.env_cfg.env_horizon = cfg.eval_env_cfg.env_horizon = env.max_episode_steps = max_episode_steps
	
	return env