from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import torch
from gymnasium.vector import VectorEnv

from mani_skill.utils import gym_utils
from mani_skill.utils.structs.types import Array

if TYPE_CHECKING:
    from mani_skill.envs.sapien_env import BaseEnv


class ManiSkillVectorEnv(VectorEnv):
    """
    Gymnasium Vector Env implementation for ManiSkill environments running on the GPU for parallel simulation and optionally parallel rendering

    Note that currently this also assumes modeling tasks as infinite horizon (e.g. terminations is always False, only reset when timelimit is reached)

    Args:
        env: The environment created via gym.make / after wrappers are applied. If a string is given, we use gym.make(env) to create an environment
        num_envs: The number of parallel environments. This is only used if the env argument is a string
        env_kwargs: Environment kwargs to pass to gym.make. This is only used if the env argument is a string
        auto_reset (bool): Whether this wrapper will auto reset the environment (following the same API/conventions as Gymnasium).
            Default is True (recommended as most ML/RL libraries use auto reset)
        ignore_terminations (bool): Whether this wrapper ignores terminations when deciding when to auto reset. Terminations can be caused by
            the task reaching a success or fail state as defined in a task's evaluation function. Default is False, meaning there is early stop in
            episode rollouts. If set to True, this would generally for situations where you may want to model a task as infinite horizon where a task
            stops only due to the timelimit.
        handle_truncations (bool): Whether this wrapper handles truncations based on the given max_episode_steps or whatever
            max_episode_steps is defined in the environment's spec. If this is True it will override any truncation values set by
            previous wrappers. Default is True.
    """

    def __init__(
        self,
        env: Union[BaseEnv, str],
        num_envs: int = None,
        auto_reset: bool = True,
        max_episode_steps: int = None,
        ignore_terminations: bool = False,
        handle_truncations: bool = True,
        **kwargs,
    ):
        if isinstance(env, str):
            self._env = gym.make(env, num_envs=num_envs, **kwargs)
        else:
            self._env = env
            num_envs = self.base_env.num_envs
        self.auto_reset = auto_reset
        self.ignore_terminations = ignore_terminations
        self.handle_truncations = handle_truncations
        super().__init__(
            num_envs, self._env.single_observation_space, self._env.single_action_space
        )

        self.returns = torch.zeros(self.num_envs, device=self.base_env.device)
        self.max_episode_steps = max_episode_steps
        if (
            self.max_episode_steps is None
            and self.base_env.spec.max_episode_steps is not None
        ):
            self.max_episode_steps = self.base_env.spec.max_episode_steps
        if self.max_episode_steps is None:
            # search wrappers to find where max episode steps may have been defined
            self.max_episode_steps = gym_utils.find_max_episode_steps_value(env)

    @property
    def device(self):
        return self.base_env.device

    @property
    def base_env(self) -> BaseEnv:
        return self._env.unwrapped

    @property
    def unwrapped(self):
        return self.base_env

    def reset(
        self,
        *,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = dict(),
    ):
        obs, info = self._env.reset(seed=seed, options=options)
        if "env_idx" in options:
            env_idx = options["env_idx"]
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.base_env.device)
            mask[env_idx] = True
            self.returns[mask] = 0
        else:
            self.returns *= 0
        return obs, info

    def step(
        self, actions: Union[Array, Dict]
    ) -> Tuple[Array, Array, Array, Array, Dict]:
        obs, rew, terminations, truncations, infos = self._env.step(actions)
        self.returns += rew

        infos["episode"] = dict(r=self.returns)

        # fix issue with gymnasium replacing truncations with a single bool
        if self.handle_truncations and self.max_episode_steps is not None:
            truncations: torch.Tensor = (
                self.base_env.elapsed_steps >= self.max_episode_steps
            )
        if isinstance(terminations, bool):
            terminations = torch.tensor([terminations], device=self.device)
        if self.ignore_terminations:
            terminations[:] = False
        dones = torch.logical_or(terminations, truncations)
        infos[
            "real_next_obs"
        ] = obs  # not part of standard API but makes some RL code slightly less complicated
        if dones.any():
            infos["episode"]["r"] = self.returns.clone()
            final_obs = obs
            env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
            obs, _ = self.reset(options=dict(env_idx=env_idx))
            infos["episode"]["_r"] = dones
            infos["final_info"] = infos.copy()
            # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
            infos["final_observation"] = final_obs
            # NOTE (stao): that adding masks like below is a bit redundant and not necessary
            # but this is to follow the standard gymnasium API
            infos["_final_info"] = dones
            infos["_final_observation"] = dones
            infos["_elapsed_steps"] = dones
            # NOTE (stao): Unlike gymnasium, the code here does not add masks for every key in the info object.
        return obs, rew, terminations, truncations, infos

    def close(self):
        return self._env.close()

    def call(self, name: str, *args, **kwargs):
        function = getattr(self.env, name)
        return function(*args, **kwargs)

    def get_attr(self, name: str):
        raise RuntimeError(
            "To get an attribute get it from the .env property of this object"
        )

    def render(self):
        return self.base_env.render()
