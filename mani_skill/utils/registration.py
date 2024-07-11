from __future__ import annotations

import json
import sys
from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING, Dict, Type

import gymnasium as gym
from gymnasium.envs.registration import EnvSpec as GymEnvSpec
from gymnasium.envs.registration import WrapperSpec

from mani_skill import logger
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

if TYPE_CHECKING:
    from mani_skill.envs.sapien_env import BaseEnv


class EnvSpec:
    def __init__(
        self,
        uid: str,
        cls: Type[BaseEnv],
        max_episode_steps=None,
        default_kwargs: dict = None,
    ):
        """A specification for a ManiSkill environment."""
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
    """Register a ManiSkill environment."""

    # hacky way to avoid circular import errors when users inherit a task in ManiSkill and try to register it themselves
    from mani_skill.envs.sapien_env import BaseEnv

    if name in REGISTERED_ENVS:
        logger.warn(f"Env {name} already registered")
    if not issubclass(cls, BaseEnv):
        raise TypeError(f"Env {name} must inherit from BaseEnv")
    REGISTERED_ENVS[name] = EnvSpec(
        name, cls, max_episode_steps=max_episode_steps, default_kwargs=default_kwargs
    )


class TimeLimitWrapper(gym.Wrapper):
    """like the standard gymnasium timelimit wrapper but fixes truncated variable to be a torch tensor and batched"""

    def __init__(self, env: gym.Env, max_episode_steps: int):
        super().__init__(env)
        prev_frame_locals = sys._getframe(1).f_locals
        frame = sys._getframe(1)
        # check for user supplied max_episode_steps during gym.make calls
        if frame.f_code.co_name == "make" and "max_episode_steps" in prev_frame_locals:
            if prev_frame_locals["max_episode_steps"] is not None:
                max_episode_steps = prev_frame_locals["max_episode_steps"]
            # do some wrapper surgery to remove the previous timelimit wrapper
            # with gymnasium 0.29.1, this will remove the timelimit wrapper and nothing else.
            curr_env = env
            while curr_env is not None:
                if isinstance(curr_env, gym.wrappers.TimeLimit):
                    self.env = curr_env.env
                    break
        self._max_episode_steps = max_episode_steps

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        truncated = self.base_env.elapsed_steps >= self._max_episode_steps
        return observation, reward, terminated, truncated, info


def make(env_id, **kwargs):
    """Instantiate a ManiSkill environment.

    Args:
        env_id (str): Environment ID.
        as_gym (bool, optional): Add TimeLimit wrapper as gym.
        **kwargs: Keyword arguments to pass to the environment.
    """
    if env_id not in REGISTERED_ENVS:
        raise KeyError("Env {} not found in registry".format(env_id))
    env_spec = REGISTERED_ENVS[env_id]
    env = env_spec.make(**kwargs)
    return env


def make_vec(env_id, **kwargs):
    env = gym.make(env_id, **kwargs)
    env = ManiSkillVectorEnv(env)
    return env


def register_env(uid: str, max_episode_steps=None, override=False, **kwargs):
    """A decorator to register ManiSkill environments.

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

        # Register for ManiSkill
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
            additional_wrappers=[
                WrapperSpec(
                    "MSTimeLimit",
                    entry_point="mani_skill.utils.registration:TimeLimitWrapper",
                    kwargs=dict(max_episode_steps=max_episode_steps),
                )
            ],
        )

        return cls

    return _register_env
