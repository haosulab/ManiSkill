from typing import Any, List, Optional, Type, Union

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv as SB3VecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from typing import Dict
from mani_skill.envs.sapien_env import BaseEnv
import torch
 

def to_cpu_numpy(data: Any) -> Any:
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, dict):
        return {key: to_cpu_numpy(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [to_cpu_numpy(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(to_cpu_numpy(item) for item in data)
    elif isinstance(data, set):
        return {to_cpu_numpy(item) for item in data}
    else:
        return data

def select_index_from_dict(data, i: int):
    if isinstance(data, dict):
        out={}
        for k, v in data.items():
            if isinstance(v, dict):
                out[k] = select_index_from_dict(v, i)
            else:
                out[k] = v[i]
        return out
    else:
        return data[i]

class ManiSkillSB3VectorEnv(SB3VecEnv):
    """A wrapper for to make ManiSkill parallel simulation compatible with SB3 VecEnv"""

    def __init__(self, env: BaseEnv):
        super().__init__(
            env.num_envs, env.single_observation_space, env.single_action_space
        )
        self._env = env

    def reset(self) -> VecEnvObs:
        obs,_=self._env.reset(seed=self.seed()[0])
        return to_cpu_numpy(obs)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions=actions

    def step_wait(self) -> VecEnvStepReturn:
        vec_obs, rews, terminations, truncations, infos = self._env.step(self.actions)
        vec_obs,rews,terminations,truncations,infos=to_cpu_numpy(vec_obs),to_cpu_numpy(rews),to_cpu_numpy(terminations),to_cpu_numpy(truncations),to_cpu_numpy(infos)
        #print(vec_obs, rews, terminations, truncations, infos)
        new_infos: List[Dict[str, Any]] = [{}] * self.num_envs
        for env_idx in range(self.num_envs):
            new_infos[env_idx]['is_success']=infos['success'][env_idx]

        for env_idx in range(self.num_envs):
            new_infos[env_idx]["TimeLimit.truncated"] = (
                truncations[env_idx] and not terminations[env_idx]
            )
        dones = terminations | truncations
        if not dones.any():
            return vec_obs, rews, dones, new_infos

        for i, done in enumerate(dones):
            if done:
                new_infos[i]["terminal_observation"] = to_cpu_numpy(select_index_from_dict(vec_obs, i))

        reset_indices = np.where(dones)[0]
        options=dict()
        options["env_idx"]=reset_indices

        vec_obs,_ = self._env.reset(options=options)
        vec_obs=to_cpu_numpy(vec_obs)
        return vec_obs, rews, dones, new_infos


    def close(self) -> None:
        return self._env.close()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        indices=self._get_indices(indices)
        return [getattr(self._env, attr_name) for i in indices]

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        setattr(self._env, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs
    ) -> List[Any]:
        indices=self._get_indices(indices)
        return [getattr(self._env, method_name)(*method_args, **method_kwargs) for i in indices]

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        return [False] * self.num_envs
    
    @property
    def device(self):
        return self.base_env.device

    @property
    def base_env(self) -> BaseEnv:
        return self._env.unwrapped

    @property
    def unwrapped(self):
        return self.base_env
    
    @property
    def env(self):
        return self._env
    
    @property
    def spec(self):
        return self._env.spec
