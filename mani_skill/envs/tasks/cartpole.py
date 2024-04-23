"""
Code for a minimal environment/task with just a robot being loaded. We recommend copying this template and modifying as you need.

At a high-level, ManiSkill2 tasks can minimally be defined by how the environment resets, what agents/objects are
loaded, goal parameterization, and success conditions

Environment reset is comprised of running two functions, `self._reconfigure` and `self.initialize_episode`, which is auto
run by ManiSkill2. As a user, you can override a number of functions that affect reconfiguration and episode initialization.

Reconfiguration will reset the entire environment scene and allow you to load/swap assets and agents.

Episode initialization will reset the positions of all objects (called actors), articulations, and agents,
in addition to initializing any task relevant data like a goal

See comments for how to make your own environment and what each required function should do
"""

from collections import OrderedDict
from typing import Any, Dict, Union

import numpy as np
import torch
import torch.random

from mani_skill.agents.robots import CartPole
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.reward.common import tolerance
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


class CartpoleEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["cart_pole"]

    agent: Union[CartPole]

    def __init__(self, *args, robot_uids="cart_pole", robot_init_qpos_noise=0.1, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        self.agent.controllers[self.agent._control_mode].balance_passive_force = False

    @property
    def _default_sim_cfg(self):
        return SimConfig(
            gpu_memory_cfg=GPUMemoryConfig(
                found_lost_pairs_capacity=2 ** 25, max_rigid_patch_count=2 ** 18
            )
        )

    @property
    def _sensor_configs(self):
        pose = sapien_utils.look_at(eye=[2, 0, 4], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _human_render_camera_configs(self):
        pose = sapien_utils.look_at([2, 2.4, 3], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        pass

    def evaluate(self):
        return dict()

    def _get_obs_extra(self, info: Dict):
        obs = OrderedDict(
            velocity=self.agent.pole_link.linear_velocity,
            angular_velocity=self.agent.pole_link.angular_velocity,

        )
        if self._obs_mode in ["state", "state_dict"]:
            pass
        return obs

    def pole_angle_cosine(self):
        # return the cosine of the pole angle, qpos[1] = 0 is upright, qpos[1] = pi / -pi is upside down
        return torch.cos(self.agent.robot.get_qpos()[:, 1])

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        cart_pos = self.agent.cart_link.pose.p[:, 0]    # (B, ), we only care about x position
        centered = tolerance(cart_pos, margin=2)
        centered = (1 + centered) / 2       # (B, )

        small_control = tolerance(action, margin=1,
                                  value_at_margin=0,
                                  sigmoid='quadratic')[:, 0]
        small_control = (4 + small_control) / 5

        angular_vel = self.agent.robot.get_qvel()[:, 1]
        small_velocity = tolerance(angular_vel, margin=5)
        small_velocity = (1 + small_velocity) / 2    # (B, )

        upright = (self.pole_angle_cosine() + 1) / 2  # (B, )

        # upright is 1 when the pole is upright, 0 when the pole is upside down
        # small_control is 1 when the action is small, 0.8 when the action is large
        # small_velocity is 1 when the angular velocity is small, 0.5 when the angular velocity is large
        # centered is 1 when the cart is centered, 0 when the cart is at the edge of the screen

        reward = (upright * centered * small_control * small_velocity).mean()
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward


@register_env("CartpoleBalance-v1", max_episode_steps=250)
class CartpoleBalanceEnv(CartpoleEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            robot_init_qpos_noise=0.05,
            **kwargs,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            dof = self.agent.robot.dof
            if isinstance(dof, torch.Tensor):
                dof = dof[0]

            init_qpos = torch.zeros(b, dof) * np.pi
            init_qpos += torch.randn((b, dof)) * self.robot_init_qpos_noise
            self.agent.reset(init_qpos)
            self.agent.robot.set_pose(
                Pose.create_from_pq(
                    torch.tensor([0.0, 0, -0.5 + 0.022]),
                    torch.tensor([1, 0, 0, 0]),
                )
            )

    def evaluate(self):
        return dict(
            fail=self.pole_angle_cosine() < 0
        )


@register_env("CartpoleSwingUp-v1", max_episode_steps=250)
class CartpoleSwingUpEnv(CartpoleEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            robot_init_qpos_noise=0.3,
            **kwargs,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            dof = self.agent.robot.dof
            if isinstance(dof, torch.Tensor):
                dof = dof[0]

            init_qpos = torch.rand(b, dof) * np.pi
            init_qpos += torch.randn((b, dof)) * self.robot_init_qpos_noise
            self.agent.reset(init_qpos)
            self.agent.robot.set_pose(
                Pose.create_from_pq(
                    torch.tensor([0.0, 0, -0.5 + 0.022]),
                    torch.tensor([1, 0, 0, 0]),
                )
            )