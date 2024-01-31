from typing import Dict, List, Optional, Tuple, Union

import gymnasium as gym
import torch
from gymnasium.vector import VectorEnv

from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.structs.types import Array


class ManiSkillVectorEnv(VectorEnv):
    """
    Gymnasium Vector Env implementation for ManiSkill environments running on the GPU for parallel simulation and optionally parallel rendering

    Note that currently this also assumes modeling tasks as infinite horizon (e.g. terminations is always False, only reset when timelimit is reached)
    """

    def __init__(
        self,
        env: Union[BaseEnv, str],
        num_envs: int = None,
        env_kwargs: Dict = dict(),
        auto_reset: bool = True,
    ):
        if isinstance(env, str):
            self._env = gym.make(env, num_envs=num_envs, **env_kwargs)
        else:
            self._env = env
            num_envs = self.base_env.num_envs
        self.auto_reset = auto_reset
        super().__init__(
            num_envs, self._env.single_observation_space, self._env.single_action_space
        )

        self.returns = torch.zeros(self.num_envs, device=self.base_env.device)
        self.max_episode_steps = self._env.spec.max_episode_steps

    @property
    def base_env(self) -> BaseEnv:
        return self._env.unwrapped

    def reset(
        self,
        *,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = None,
    ):
        obs, info = self._env.reset(seed=seed, options=options)
        self.returns *= 0
        return obs, info

    def step(
        self, actions: Union[Array, Dict]
    ) -> Tuple[Array, Array, Array, Array, Dict]:
        obs, rew, _, _, infos = self._env.step(actions)
        self.returns += rew
        infos["episode"] = dict(r=self.returns)
        truncations: torch.Tensor = (
            self.base_env.elapsed_steps >= self.max_episode_steps
        )
        terminations = torch.zeros(self.num_envs, device=self.base_env.device)
        if truncations.any():
            # TODO (stao): permit reset by indicies later
            infos["episode"]["r"] = self.returns.clone()
            final_obs = obs
            obs, _ = self.reset()
            new_infos = dict()
            new_infos["final_info"] = infos
            new_infos["final_observation"] = final_obs
            infos = new_infos
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
