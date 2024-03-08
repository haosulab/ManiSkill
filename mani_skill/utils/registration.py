import json
from copy import deepcopy
from functools import partial
from typing import Dict, Type

import gymnasium as gym
from gymnasium.envs.registration import EnvSpec as GymEnvSpec

from mani_skill import logger
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


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


def make(env_id, enable_segmentation=False, **kwargs):
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

    env = env_spec.make(**kwargs)
    return env


def make_vec(env_id, **kwargs):
    max_episode_steps = kwargs.pop("max_episode_steps", None)
    env = make(env_id, **kwargs)
    env = ManiSkillVectorEnv(env, max_episode_steps=max_episode_steps)
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
    try:
        json.dumps(kwargs)
    except TypeError:
        raise RuntimeError(
            f"You cannot register_env with non json dumpable kwargs, e.g. classes or types. If you really need to do this, it is recommended to create a mapping of string to the unjsonable data and to pass the string in the kwarg and during env creation find the data you need"
        )

    def _register_env(cls):
        if uid in REGISTERED_ENVS:
            if override:
                from gymnasium.envs.registration import registry

                logger.warn(f"Override registered env {uid}")
                REGISTERED_ENVS.pop(uid)
                registry.pop(uid)
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
            entry_point=partial(make, env_id=uid),
            vector_entry_point=partial(make_vec, env_id=uid),
            max_episode_steps=max_episode_steps,
            disable_env_checker=True,  # Temporary solution as we allow empty observation spaces
            kwargs=deepcopy(kwargs),
        )

        return cls

    return _register_env
