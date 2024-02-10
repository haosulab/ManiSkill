"""Wrapper for limiting the time steps of an environment. Adapted from Gynmnasium to flexibly handle GPU simulated tasks"""
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import gymnasium as gym
import torch

from mani_skill2.envs.sapien_env import BaseEnv

if TYPE_CHECKING:
    from gymnasium.envs.registration import EnvSpec


class TimeLimit(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        max_episode_steps: int,
    ):
        """Initializes the :class:`TimeLimit` wrapper with an environment and the number of steps after which truncation will occur.

        Args:
            env: The environment to apply the wrapper
            max_episode_steps: An optional max episode steps (if ``None``, ``env.spec.max_episode_steps`` is used)
        """
        gym.Wrapper.__init__(self, env)

        self._max_episode_steps = max_episode_steps

    @property
    def _base_env(self) -> BaseEnv:
        return self.env.unwrapped

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        truncated = self._base_env.elapsed_steps >= self._max_episode_steps
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

    @property
    def spec(self) -> EnvSpec | None:
        """Modifies the environment spec to include the `max_episode_steps=self._max_episode_steps`."""
        if self._cached_spec is not None:
            return self._cached_spec

        env_spec = self.env.spec
        if env_spec is not None:
            env_spec = deepcopy(env_spec)
            env_spec.max_episode_steps = self._max_episode_steps

        self._cached_spec = env_spec
        return env_spec
