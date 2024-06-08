import gymnasium as gym
import sapien.physx as physx

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common


class ManiSkillCPUGymWrapper(gym.Wrapper):
    """This wrapper wraps any maniskill env created via gym.make to ensure the outputs of
    env.render, env.reset, env.step are all numpy arrays and are not batched. This is only useful
    for use with the CPU simulation backend of ManiSkill. This wrapper should generally be applied after
    wrappers as most wrappers for ManiSkill assume data returned is batched and is a torch tensor"""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert (
            self.base_env.num_envs == 1
        ), "This wrapper is only for environments without parallelization"
        assert (
            not physx.is_gpu_enabled()
        ), "This wrapper is only for environments on the CPU backend"
        self.observation_space = self.base_env.single_observation_space
        self.action_space = self.base_env.single_action_space

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return common.unbatch(
            common.to_numpy(obs),
            common.to_numpy(reward),
            common.to_numpy(terminated),
            common.to_numpy(truncated),
            common.to_numpy(info),
        )

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        return common.unbatch(common.to_numpy(obs), common.to_numpy(info))

    def render(self):
        ret = self.env.render()
        if self.render_mode in ["rgb_array", "sensors"]:
            return common.unbatch(common.to_numpy(ret))
