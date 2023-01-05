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
        uuid: str,
        cls: Type[BaseEnv],
        max_episode_steps=None,
        default_kwargs: dict = None,
    ):
        """A specification for a ManiSkill2 environment."""
        self.uuid = uuid
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
            self.uuid,
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


def make(env_id, from_gym=False, **kwargs):
    """Instantiate a ManiSkill2 environment."""
    if env_id not in REGISTERED_ENVS:
        raise KeyError("Env {} not found in registry".format(env_id))

    # Dispatch observation mode
    obs_mode = kwargs.get("obs_mode")
    if obs_mode not in ["state", "state_dict", "none"]:
        kwargs["obs_mode"] = "image"

    # Whether to enable robot segmentation
    enable_robot_seg = kwargs.get("enable_robot_seg", False)
    if enable_robot_seg:
        camera_cfgs = kwargs.get("camera_cfgs", {})
        camera_cfgs["add_segmentation"] = True
        kwargs["camera_cfgs"] = camera_cfgs

    env_spec = REGISTERED_ENVS[env_id]
    env = env_spec.make(**kwargs)

    if enable_robot_seg:
        env = RobotSegmentationObservationWrapper(env)

    # Dispatch observation wrapper
    if obs_mode == "rgbd":
        env = RGBDObservationWrapper(env)
        env.obs_mode = obs_mode
    elif obs_mode == "pointcloud":
        env = PointCloudObservationWrapper(env)
        env.obs_mode = obs_mode

    # To make it compatible with gym
    if not from_gym:
        env.unwrapped.spec = env_spec.gym_spec
        if env_spec.max_episode_steps is not None:
            env = gym.wrappers.TimeLimit(
                env, max_episode_steps=env_spec.max_episode_steps
            )

    return env


def register_env(uuid: str, max_episode_steps=None, **kwargs):
    """A decorator to register ManiSkill2 environments.

    Args:
        uuid (str): unique id of the environment.

    Notes:
        - `max_episode_steps` is processed differently from other keyword arguments in gym.
          `gym.make` wraps the env with `gym.wrappers.TimeLimit` to limit the maximum number of steps.
        - `gym.EnvSpec` uses kwargs instead of **kwargs!
    """

    def _register_env(cls):
        # Register for ManiSkil2
        register(
            uuid,
            cls,
            max_episode_steps=max_episode_steps,
            default_kwargs=deepcopy(kwargs),
        )

        # Register for gym
        gym.register(
            uuid,
            entry_point=partial(make, env_id=uuid, from_gym=True),
            max_episode_steps=max_episode_steps,
            kwargs=deepcopy(kwargs),
        )

        return cls

    return _register_env
