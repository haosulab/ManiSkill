from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import torch
from gymnasium.vector import VectorEnv

from mani_skill.utils.structs.types import Array
from mani_skill.utils.common import torch_clone_dict

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
        record_metrics (bool): If True, the returned info objects will contain the metrics: return, length, success_once, success_at_end, fail_once, fail_at_end.
            success/fail metrics are recorded only when the environment has success/fail criteria. success/fail_at_end are recorded only when ignore_terminations is True.
    """

    def __init__(
        self,
        env: Union[BaseEnv, str],
        num_envs: int = None,
        auto_reset: bool = True,
        ignore_terminations: bool = False,
        record_metrics: bool = False,
        **kwargs,
    ):
        if isinstance(env, str):
            self._env = gym.make(env, num_envs=num_envs, **kwargs)
        else:
            self._env = env
            num_envs = self.base_env.num_envs
        self.auto_reset = auto_reset
        self.ignore_terminations = ignore_terminations
        self.record_metrics = record_metrics
        self.spec = self._env.spec
        super().__init__(
            num_envs,
            self._env.get_wrapper_attr("single_observation_space"),
            self._env.get_wrapper_attr("single_action_space"),
        )
        if not self.ignore_terminations and auto_reset:
            assert (
                self.base_env.reconfiguration_freq == 0 or self.base_env.num_envs == 1
            ), "With partial resets, environment cannot be reconfigured automatically"

        if self.record_metrics:
            self.success_once = torch.zeros(
                self.num_envs, device=self.base_env.device, dtype=torch.bool
            )
            self.fail_once = torch.zeros(
                self.num_envs, device=self.base_env.device, dtype=torch.bool
            )
            self.returns = torch.zeros(
                self.num_envs, device=self.base_env.device, dtype=torch.float32
            )

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
            if self.record_metrics:
                self.success_once[mask] = False
                self.fail_once[mask] = False
                self.returns[mask] = 0
        else:
            if self.record_metrics:
                self.success_once[:] = False
                self.fail_once[:] = False
                self.returns[:] = 0
        return obs, info

    def step(
        self, actions: Union[Array, Dict]
    ) -> Tuple[Array, Array, Array, Array, Dict]:
        obs, rew, terminations, truncations, infos = self._env.step(actions)

        if self.record_metrics:
            episode_info = dict()
            self.returns += rew
            if "success" in infos:
                self.success_once = self.success_once | infos["success"]
                episode_info["success_once"] = self.success_once.clone()
            if "fail" in infos:
                self.fail_once = self.fail_once | infos["fail"]
                episode_info["fail_once"] = self.fail_once.clone()
            episode_info["return"] = self.returns.clone()
            episode_info["episode_len"] = self.base_env.elapsed_steps.clone()
            episode_info["reward"] = (
                episode_info["return"] / episode_info["episode_len"]
            )

        if isinstance(terminations, bool):
            terminations = torch.tensor([terminations], device=self.device)

        if self.ignore_terminations:
            terminations[:] = False
            if self.record_metrics:
                if "success" in infos:
                    episode_info["success_at_end"] = infos["success"].clone()
                if "fail" in infos:
                    episode_info["fail_at_end"] = infos["fail"].clone()
        if self.record_metrics:
            infos["episode"] = episode_info

        dones = torch.logical_or(terminations, truncations)

        if dones.any() and self.auto_reset:
            final_obs = torch_clone_dict(obs)
            env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
            final_info = torch_clone_dict(infos)
            obs, infos = self.reset(options=dict(env_idx=env_idx))
            # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
            infos["final_observation"] = final_obs
            infos["final_info"] = final_info
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
        function = getattr(self._env, name)
        return function(*args, **kwargs)

    def get_attr(self, name: str):
        raise RuntimeError(
            "To get an attribute get it from the .env property of this object"
        )

    def render(self):
        return self.base_env.render()
