"""Adapted from https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/cartpole.py"""
import os
from typing import Any, Dict, Union

import numpy as np
import torch

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization, rewards
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import (
    Array,
    GPUMemoryConfig,
    SceneConfig,
    SimConfig,
)

MJCF_FILE = f"{os.path.join(os.path.dirname(__file__), 'assets/cartpole.xml')}"


class CartPoleRobot(BaseAgent):
    uid = "cart_pole"
    mjcf_path = MJCF_FILE
    disable_self_collisions = True

    @property
    def _controller_configs(self):
        # NOTE it is impossible to copy joint properties from original xml files, have to tune manually until
        # it looks approximately correct
        pd_joint_delta_pos = PDJointPosControllerConfig(
            ["slider"],
            -1,
            1,
            damping=200,
            stiffness=2000,
            use_delta=True,
        )
        rest = PassiveControllerConfig(["hinge_1"], damping=0, friction=0)
        return dict(
            pd_joint_delta_pos=dict(
                slider=pd_joint_delta_pos, rest=rest, balance_passive_force=False
            )
        )

    def _load_articulation(self):
        """
        Load the robot articulation
        """
        loader = self.scene.create_mjcf_loader()
        asset_path = str(self.mjcf_path)

        loader.name = self.uid

        # only need the robot
        self.robot = loader.parse(asset_path)[0][0].build()
        assert self.robot is not None, f"Fail to load URDF/MJCF from {asset_path}"

        # Cache robot link ids
        self.robot_link_ids = [link.name for link in self.robot.get_links()]


# @register_env("MS-CartPole-v1", max_episode_steps=500)
# class CartPoleEnv(BaseEnv):
#     SUPPORTED_REWARD_MODES = ["sparse", "none"]

#     SUPPORTED_ROBOTS = [CartPoleRobot]
#     agent: Union[CartPoleRobot]

#     CART_RANGE = [-0.25, 0.25]
#     ANGLE_COSINE_RANGE = [0.995, 1]

#     def __init__(self, *args, robot_uids=CartPoleRobot, **kwargs):
#         super().__init__(*args, robot_uids=robot_uids, **kwargs)


class CartpoleEnv(BaseEnv):

    agent: Union[CartPoleRobot]

    def __init__(self, *args, robot_uids=CartPoleRobot, **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            spacing=20,
            scene_cfg=SceneConfig(
                solver_position_iterations=4, solver_velocity_iterations=0
            ),
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0, -4, 1], target=[0, 0, 1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0, -4, 1], target=[0, 0, 1])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        loader = self.scene.create_mjcf_loader()
        articulation_builders, actor_builders, sensor_configs = loader.parse(MJCF_FILE)
        for a in actor_builders:
            a.build(a.name)

    def evaluate(self):
        return dict()

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            velocity=self.agent.robot.links_map["pole_1"].linear_velocity,
            angular_velocity=self.agent.robot.links_map["pole_1"].angular_velocity,
        )
        return obs

    @property
    def pole_angle_cosine(self):
        return torch.cos(self.agent.robot.joints_map["hinge_1"].qpos)

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        cart_pos = self.agent.robot.links_map["cart"].pose.p[
            :, 0
        ]  # (B, ), we only care about x position
        centered = rewards.tolerance(cart_pos, margin=2)
        centered = (1 + centered) / 2  # (B, )

        small_control = rewards.tolerance(
            action, margin=1, value_at_margin=0, sigmoid="quadratic"
        )[:, 0]
        small_control = (4 + small_control) / 5

        angular_vel = self.agent.robot.get_qvel()[:, 1]
        small_velocity = rewards.tolerance(angular_vel, margin=5)
        small_velocity = (1 + small_velocity) / 2  # (B, )

        upright = (self.pole_angle_cosine + 1) / 2  # (B, )

        # upright is 1 when the pole is upright, 0 when the pole is upside down
        # small_control is 1 when the action is small, 0.8 when the action is large
        # small_velocity is 1 when the angular velocity is small, 0.5 when the angular velocity is large
        # centered is 1 when the cart is centered, 0 when the cart is at the edge of the screen

        reward = upright * centered * small_control * small_velocity
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward


@register_env("MS-CartpoleBalance-v1", max_episode_steps=1000)
class CartpoleBalanceEnv(CartpoleEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            qpos = torch.zeros((b, 2))
            qpos[:, 0] = randomization.uniform(-0.1, 0.1, size=(b,))
            qpos[:, 1] = randomization.uniform(-0.034, 0.034, size=(b,))
            qvel = torch.randn(size=(b, 2)) * 0.01
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_qvel(qvel)

    def evaluate(self):
        return dict(fail=self.pole_angle_cosine < 0)


@register_env("MS-CartpoleSwingUp-v1", max_episode_steps=1000)
class CartpoleSwingUpEnv(CartpoleEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            qpos = torch.zeros((b, 2))
            qpos[:, 0] = torch.randn((b,)) * 0.01
            qpos[:, 1] = torch.randn((b,)) * 0.01 + torch.pi
            qvel = torch.randn(size=(b, 2)) * 0.01
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_qvel(qvel)
