"""
Gymnasium environment interface for controlling a real robot.

The setup of this environment is the same as the ManiSkill BaseEnv in terms of observation and action spaces, and uses the same
controller code for the real robot as used in simulation.

Note that as many operations of simulation environments are not available for real environments, we do not inherit from BaseEnv and simply
reference some of the BaseEnv functions for consistency instead.

One small difference as well between RealEnv and BaseEnv is that the code for fetching raw real world sensor data is robot dependent, not environment dependent.

So for real world deployments you may take an existing implementation of a real robot class and use it as a starting point for your own implementation to add e.g. more cameras
or generate other kinds of sensor data.
"""
from typing import Any, Dict, Optional

import gymnasium as gym
import torch

from mani_skill.agents.base_real_agent import BaseRealAgent
from mani_skill.envs.sapien_env import BaseEnv


class RealEnv(gym.Env):
    def __init__(
        self,
        obs_mode: Optional[str] = None,
        reward_mode: Optional[str] = None,
        control_mode: Optional[str] = None,
        render_mode: Optional[str] = None,
        robot_uids: BaseRealAgent = None,
    ):

        # create the real robot objects if needed
        self.agent = robot_uids

    def _step_action(self, action):
        pass

    def step(self, action):
        return BaseEnv.step(self, action)
        self._step_action(action)
        self.elapsed_steps += 1
        obs = self.get_obs()
        return (
            obs,
            None,  # reward
            False,  # terminated
            False,  # truncated
            dict(),  # info
        )

    def get_info(self):
        return BaseEnv.get_info(self)

    def reset(self, seed=0, options=None):
        pass

    def render(self):
        return BaseEnv.render(self)

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """
        Computes the sparse reward. By default this function tries to use the success/fail information in
        returned by the evaluate function and gives +1 if success, -1 if fail, 0 otherwise"""
        return BaseEnv.compute_sparse_reward(self, obs, action, info)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        raise NotImplementedError()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        raise NotImplementedError()
