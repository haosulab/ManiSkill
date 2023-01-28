from copy import deepcopy
from functools import partial
from typing import Dict, Type

import gym
from gym.envs.registration import EnvSpec as GymEnvSpec

from mani_skill2 import logger
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.wrappers.observation import (
    PointCloudObservationWrapper,
    RGBDObservationWrapper,
    RobotSegmentationObservationWrapper,
)


class EnvSpec:
    def __init__(
        self,
        uid: str,
        cls: Type[BaseEnv],
        max_episode_steps=None,
        default_kwargs: dict = None,
    ):
        """A specification for a ManiSkill2 environment."""
        self.uid = uid
        self.cls = cls
        self.max_episode_steps = max_episode_steps
        self.default_kwargs = {} if default_kwargs is None else default_kwargs

    def make(self, **kwargs):
        _kwargs = self.default_kwargs.copy()
        _kwargs.update(kwargs)
        return self.cls(**_kwargs)

    @property
    def gym_spec(self):
        """Return a gym EnvSpec for this env"""
        entry_point = self.cls.__module__ + ":" + self.cls.__name__
        return GymEnvSpec(
            self.uid,
            entry_point,
            max_episode_steps=self.max_episode_steps,
            kwargs=self.default_kwargs,
        )


REGISTERED_ENVS: Dict[str, EnvSpec] = {}


def register(
    name: str, cls: Type[BaseEnv], max_episode_steps=None, default_kwargs: dict = None
):
    """Register a ManiSkill2 environment."""
    if name in REGISTERED_ENVS:
        logger.warn(f"Env {name} already registered")
    if not issubclass(cls, BaseEnv):
        raise TypeError(f"Env {name} must inherit from BaseEnv")
    REGISTERED_ENVS[name] = EnvSpec(
        name, cls, max_episode_steps=max_episode_steps, default_kwargs=default_kwargs
    )


def make(env_id, as_gym=True, enable_segmentation=False, **kwargs):
    """Instantiate a ManiSkill2 environment.

    Args:
        env_id (str): Environment ID.
        as_gym (bool, optional): Add TimeLimit wrapper as gym.
        enable_segmentation (bool, optional): Whether to include Segmentation in observations.
        **kwargs: Keyword arguments to pass to the environment.
    """
    if env_id not in REGISTERED_ENVS:
        raise KeyError("Env {} not found in registry".format(env_id))
    env_spec = REGISTERED_ENVS[env_id]

    # Dispatch observation mode
    obs_mode = kwargs.get("obs_mode")
    if obs_mode is None:
        obs_mode = env_spec.cls.SUPPORTED_OBS_MODES[0]
    if obs_mode not in ["state", "state_dict", "none", "particles"]:
        kwargs["obs_mode"] = "image"

    # Add segmentation texture
    if "robot_seg" in obs_mode:
        enable_segmentation = True
    if enable_segmentation:
        camera_cfgs = kwargs.get("camera_cfgs", {})
        camera_cfgs["add_segmentation"] = True
        kwargs["camera_cfgs"] = camera_cfgs

    env = env_spec.make(**kwargs)

    # Dispatch observation wrapper
    if "rgbd" in obs_mode:
        env = RGBDObservationWrapper(env)
    elif "pointcloud" in obs_mode:
        env = PointCloudObservationWrapper(env)

    # Add robot segmentation wrapper
    if "robot_seg" in obs_mode:
        env = RobotSegmentationObservationWrapper(env)

    # Set observation mode on the wrapper
    if isinstance(env, gym.Wrapper):
        env.obs_mode = obs_mode

    # Compatible with gym.make
    if as_gym:
        env.unwrapped.spec = env_spec.gym_spec
        if env_spec.max_episode_steps is not None:
            env = gym.wrappers.TimeLimit(
                env, max_episode_steps=env_spec.max_episode_steps
            )

    return env


def register_env(uid: str, max_episode_steps=None, override=False, **kwargs):
    """A decorator to register ManiSkill2 environments.

    Args:
        uid (str): unique id of the environment.
        max_episode_steps (int): maximum number of steps in an episode.
        override (bool): whether to override the environment if it is already registered.

    Notes:
        - `max_episode_steps` is processed differently from other keyword arguments in gym.
          `gym.make` wraps the env with `gym.wrappers.TimeLimit` to limit the maximum number of steps.
        - `gym.EnvSpec` uses kwargs instead of **kwargs!
    """

    def _register_env(cls):
        if uid in REGISTERED_ENVS:
            if override:
                from gym.envs.registration import registry

                logger.warn(f"Override registered env {uid}")
                REGISTERED_ENVS.pop(uid)
                registry.env_specs.pop(uid)
            else:
                logger.warn(f"Env {uid} is already registered. Skip registration.")
                return cls

        # Register for ManiSkil2
        register(
            uid,
            cls,
            max_episode_steps=max_episode_steps,
            default_kwargs=deepcopy(kwargs),
        )

        # Register for gym
        gym.register(
            uid,
            entry_point=partial(make, env_id=uid, as_gym=False),
            max_episode_steps=max_episode_steps,
            kwargs=deepcopy(kwargs),
        )

        return cls

    return _register_env
