import multiprocessing as mp
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union

import gym
import numpy as np
from mani_skill2.vector.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv as SB3VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)

class ManiskillVecEnvToSB3VecEnv(SB3VecEnv):
    """
    A Stable Baselines 3 VecEnv wrapper to convert a ManiskillVecEnv into one compatible with SB3 
    """

    def __init__(self, venv: VecEnv, max_time_steps = 100, start_method: Optional[str] = None):
        SB3VecEnv.__init__(self, venv.num_envs, venv.observation_space, venv.action_space)
        self.venv = venv
        self.max_time_steps = max_time_steps
    def step_async(self, actions: np.ndarray) -> None:
        return self.venv.step_async(actions)

    def step_wait(self) -> VecEnvStepReturn:
        vec_obs, rews, dones, infos =  self.venv.step_wait()
        # handle auto reset here. Assumption is none of the vectorized envs exit early, 
        elapsed_steps = infos[0]['elapsed_steps']
        past_timelimit= elapsed_steps >= self.max_time_steps
        for i in range(len(infos)):
        
            info = infos[i]
            info['is_success'] = False
            if dones[i]:
                # set is_success to True if there is an actual success and let SB3 know
                info['is_success'] = True     
            dones[i] = False
            if past_timelimit:
                dones[i] = True
                # treat as infinite horizons. If not succesful, then we truncated it
                info["TimeLimit.truncated"] = not info['is_success'] # treat as infinite horizon
                def select_index_from_dict(data, i):
                    out = dict()
                    for k in data:
                        if isinstance(data[k], dict):
                            out[k] = select_index_from_dict(data[k], i)
                        else:
                            out[k] = data[k][i]
                info['terminal_observation'] = select_index_from_dict(vec_obs, i)
        if past_timelimit: vec_obs = self.reset()
        return vec_obs, rews, dones, infos
    def step(self, actions):
        self.step_async(actions)
        return self.venv.step_wait()
    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        return self.venv.seed(seed)

    def reset(self) -> VecEnvObs:
        return self.venv.reset()
    def close(self) -> None:
        return self.venv.close()

    # TODO stao: once we are able to render with ManiSkillVecEnv, we can add this back in
    # def get_images(self) -> Sequence[np.ndarray]:
    #     for pipe in self.remotes:
    #         # gather images from subprocesses
    #         # `mode` will be taken into account later
    #         pipe.send(("render", "rgb_array"))
    #     imgs = [pipe.recv() for pipe in self.remotes]
    #     return imgs

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        return self.venv.get_attr(attr_name, indices)

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        return self.venv.set_attr(attr_name, value, indices)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        return self.venv.env_method(method_name, *method_args, **method_kwargs)

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_remotes = self._get_target_remotes(indices)
        return [True for _ in target_remotes]

    def _get_target_remotes(self, indices: VecEnvIndices) -> List[Any]:
        return self.venv._get_target_remotes(indices)