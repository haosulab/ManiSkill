import time
from typing import Any, List, Optional, Type, Union

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env.base_vec_env import VecEnv as SB3VecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)

from mani_skill.envs.sapien_env import BaseEnv


def select_index_from_dict(data: Union[dict, np.ndarray], i: int):
    if isinstance(data, np.ndarray):
        return data[i]
    elif isinstance(data, dict):
        out = dict()
        for k, v in data.items():
            out[k] = select_index_from_dict(v, i)
        return out
    else:
        return data[i]


# TODO
class ManiSkillSB3VectorEnv(SB3VecEnv):
    """A wrapper for to make ManiSkill parallel simulation compatible with SB3 VecEnv and auto adds the monitor wrapper"""

    def __init__(self, env: BaseEnv):
        super().__init__(
            env.num_envs, env.single_observation_space, env.single_action_space
        )
        self._env = env
        self._last_seed = None
        self.t_start = time.time()
        self.episode_returns: torch.Tensor = torch.zeros(
            self.num_envs, device=self.base_env.device
        )
        self.episode_lengths: torch.Tensor = torch.zeros(
            self.num_envs, device=self.base_env.device
        )
        self.episode_times: torch.Tensor = torch.zeros(
            self.num_envs, device=self.base_env.device
        )
        self.total_steps = 0

    @property
    def base_env(self) -> BaseEnv:
        return self._env.unwrapped

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        self._last_seed = seed

    def reset(self) -> VecEnvObs:
        self.episode_returns = torch.zeros(self.num_envs, device=self.base_env.device)
        self.episode_lengths = torch.zeros(self.num_envs, device=self.base_env.device)
        obs = self._env.reset(seed=self._last_seed)[0]
        self._last_seed = None  # use seed from call to seed() once
        return obs.cpu().numpy()  # currently SB3 only support numpy arrays

    def step_async(self, actions: np.ndarray) -> None:
        self.last_actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        vec_obs, rews, terminations, truncations, infos = self._env.step(
            self.last_actions
        )

        self.episode_returns += rews
        self.episode_lengths += 1
        dones = terminations | truncations

        # Convert observations, rewards, and dones to numpy arrays
        vec_obs = vec_obs.cpu().numpy()
        rews = rews.cpu().numpy()
        dones = dones.cpu().numpy()
        new_infos = []

        for env_idx in range(self.num_envs):
            new_info = {
                "TimeLimit.truncated": truncations[env_idx]
                and not terminations[env_idx],
                "episode": {
                    "r": self.episode_returns.cpu().numpy(),
                    "l": self.episode_lengths.cpu().numpy(),
                },
            }
            # , "t": round(time.time() - self.t_start, 6)}
            new_infos.append(new_info)
        # Store terminal observations
        for i, done in enumerate(dones):
            if done:
                if "success" in infos:
                    new_infos[i]["is_success"] = infos["success"][i].cpu().numpy()
                new_infos[i]["terminal_observation"] = select_index_from_dict(
                    vec_obs, i
                )

        # Reset environments where episodes have ended
        if dones.any():
            reset_indices = np.where(dones)[0]
            new_obs = self._env.reset(options=dict(env_idx=reset_indices))[0]
            vec_obs[reset_indices] = new_obs[reset_indices].cpu().numpy()
            self.episode_returns[reset_indices] = 0
            self.episode_lengths[reset_indices] = 0

        return vec_obs, rews, dones, new_infos

    def close(self) -> None:
        return self._env.close()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        return self._env.get_attr(attr_name, indices)

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        return self._env.set_attr(attr_name, value, indices)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs
    ) -> List[Any]:
        return self._env.env_method(
            method_name, *method_args, indices=indices, **method_kwargs
        )

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        return [False] * self.num_envs
