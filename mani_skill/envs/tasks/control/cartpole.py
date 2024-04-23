"""Adapted from https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/cartpole.py"""
import os
from typing import Any, Dict, Union

import numpy as np
import torch

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import SceneConfig, SimConfig

MJCF_FILE = f"{os.path.join(os.path.dirname(__file__), 'assets/cartpole.xml')}"


class CartPoleRobot(BaseAgent):
    uid = "cartpole"
    mjcf_path = MJCF_FILE

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


@register_env("MS-CartPole-v1", max_episode_steps=500)
class CartPoleEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["sparse", "none"]

    SUPPORTED_ROBOTS = [CartPoleRobot]
    agent: Union[CartPoleRobot]

    CART_RANGE = [-0.25, 0.25]
    ANGLE_COSINE_RANGE = [0.995, 1]

    def __init__(self, *args, robot_uids=CartPoleRobot, **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=100, control_freq=100, scene_cfg=SceneConfig(solver_iterations=2)
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
        loader = self._scene.create_mjcf_loader()
        articulation_builders, actor_builders, sensor_configs = loader.parse(MJCF_FILE)
        for a in actor_builders:
            a.build(a.name)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            qpos = torch.zeros((b, 2))
            qpos[:, 0] = randomization.uniform(-0.1, 0.1, size=(b,))
            qpos[:, 1] = randomization.uniform(-0.034, 0.034, size=(b,))
            qvel = torch.randn(size=(b, 2)) * 0.01
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_qvel(qvel)

    @property
    def pole_angle_cosine(self):
        return torch.cos(self.agent.robot.joints_map["hinge_1"].qpos)

    def evaluate(self):
        cart_pos = self.agent.robot.joints_map["slider"].qpos
        pole_angle_cosine = self.pole_angle_cosine
        cart_in_bounds = cart_pos < self.CART_RANGE[1]
        cart_in_bounds = cart_in_bounds & (cart_pos > self.CART_RANGE[0])
        angle_in_bounds = pole_angle_cosine < self.ANGLE_COSINE_RANGE[1]
        angle_in_bounds = angle_in_bounds & (
            pole_angle_cosine > self.ANGLE_COSINE_RANGE[0]
        )
        return {"cart_in_bounds": cart_in_bounds, "angle_in_bounds": angle_in_bounds}

    def _get_obs_extra(self, info: Dict):
        return dict()

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return info["cart_in_bounds"] * info["angle_in_bounds"]


@register_env("CartPoleSwingUp-v1", max_episode_steps=500, override=True)
class CartPoleSwingUpEnv(CartPoleEnv):
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            qpos = torch.zeros((b, 2))
            qpos[:, 0] = 0.01 * torch.randn(size=(b,))
            qpos[:, 1] = torch.pi + 0.01 * torch.randn(size=(b,))
            qvel = torch.randn(size=(b, 2)) * 0.01
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_qvel(qvel)
            # Note DM-Control sets some randomness to other qpos values but am not sure what they are
            # as cartpole.xml seems to only load two joints
