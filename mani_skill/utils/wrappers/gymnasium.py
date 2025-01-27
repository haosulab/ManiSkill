import gymnasium as gym
import numpy as np
import sapien.physx as physx
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common


class CPUGymWrapper(gym.Wrapper):
    """This wrapper wraps any maniskill env created via gym.make to ensure the outputs of
    env.render, env.reset, env.step are all numpy arrays and are not batched.
    Essentially ensuring the environment conforms entirely to the standard gymnasium API https://gymnasium.farama.org/api/env/.
    The wrapper also optionally records standardized evaluation metrics like return and success.

    This wrapper should generally be applied after all other
    wrappers as most wrappers for ManiSkill assume data returned is a batched torch tensor

    Args:
        env (gym.Env): The environment to wrap.
        ignore_terminations (bool): If True, the environment will ignore termination signals
            and continue running until truncation. Default is False.
        record_metrics (bool): If True, the returned info objects will contain the metrics: return, length, success_once, success_at_end, fail_once, fail_at_end.
            success/fail metrics are recorded only when the environment has success/fail criteria. success/fail_at_end are recorded only when ignore_terminations is True.

    """

    def __init__(
        self,
        env: gym.Env,
        ignore_terminations: bool = False,
        record_metrics: bool = False,
    ):
        super().__init__(env)
        assert (
            self.base_env.num_envs == 1
        ), "This wrapper is only for environments without parallelization"
        assert (
            not self.base_env.gpu_sim_enabled
        ), "This wrapper is only for environments on the CPU backend"
        self.observation_space = self.base_env.single_observation_space
        self.action_space = self.base_env.single_action_space
        self.ignore_terminations = ignore_terminations
        self.record_metrics = record_metrics

        if self.record_metrics:
            self.success_once = False
            self.fail_once = False
            self.returns = []

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    def step(self, action):
        action = common.to_numpy(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = common.to_numpy(reward)
        info = common.to_numpy(info)
        if self.record_metrics:
            episode_info = dict()
            self.returns.append(reward)
            if "success" in info:
                self.success_once = self.success_once | info["success"]
                episode_info["success_once"] = self.success_once
            if "fail" in info:
                self.fail_once = self.fail_once | info["fail"]
                episode_info["fail_once"] = self.fail_once
            episode_info["return"] = np.sum(self.returns)
            episode_info["episode_len"] = len(self.returns)
            episode_info["reward"] = (
                episode_info["return"] / episode_info["episode_len"]
            )
        if self.ignore_terminations:
            terminated = False
            if self.record_metrics:
                if "success" in info:
                    episode_info["success_at_end"] = info["success"]
                if "fail" in info:
                    episode_info["fail_at_end"] = info["fail"]

        if self.record_metrics:
            info["episode"] = episode_info
        return common.unbatch(
            common.to_numpy(obs),
            reward,
            common.to_numpy(terminated),
            common.to_numpy(truncated),
            info,
        )

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if self.record_metrics:
            self.success_once = False
            self.fail_once = False
            self.returns = []
        return common.unbatch(common.to_numpy(obs), common.to_numpy(info))

    def render(self):
        ret = self.env.render()
        if self.render_mode in ["rgb_array", "sensors", "all"]:
            return common.unbatch(common.to_numpy(ret))
