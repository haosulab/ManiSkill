from typing import Any, List, Optional, Sequence, Tuple, Type, Union

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv as SB3VecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)

from mani_skill2.vector.vec_env import VecEnv


def select_index_from_dict(data: dict, i: int):
    out = dict()
    for k, v in data.items():
        if isinstance(v, dict):
            out[k] = select_index_from_dict(v, i)
        else:
            out[k] = v[i]
    return out


class SB3VecEnvWrapper(SB3VecEnv):
    """A wrapper for ManiSkill2 VecEnv to make it compatible with SB3 VecEnv"""

    def __init__(self, venv: VecEnv):
        super().__init__(
            venv.num_envs, venv.single_observation_space, venv.single_action_space
        )
        self.venv = venv
        self._last_seed = None

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        self._last_seed = seed

    def reset(self) -> VecEnvObs:
        obs = self.venv.reset(seed=self._last_seed)[0]
        self._last_seed = None  # use seed from call to seed() once
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        return self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        vec_obs, rews, terminations, truncations, infos = self.venv.step_wait()
        for env_idx in range(self.num_envs):
            infos[env_idx]["TimeLimit.truncated"] = (
                truncations[env_idx] and not terminations[env_idx]
            )
        dones = terminations | truncations
        if not dones.any():
            return vec_obs, rews, dones, infos

        for i, done in enumerate(dones):
            if done:
                # NOTE: ensure that it will not be inplace modified when reset
                infos[i]["terminal_observation"] = select_index_from_dict(vec_obs, i)

        reset_indices = np.where(dones)[0]

        vec_obs = self.venv.reset(indices=reset_indices)[0]
        return vec_obs, rews, dones, infos

    def close(self) -> None:
        return self.venv.close()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        return self.venv.get_attr(attr_name, indices)

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        return self.venv.set_attr(attr_name, value, indices)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs
    ) -> List[Any]:
        return self.venv.env_method(
            method_name, *method_args, indices=indices, **method_kwargs
        )

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        return [False] * self.num_envs
