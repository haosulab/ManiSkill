"""Adapted from https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/hopper.py"""

import os
from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization, rewards
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.control.planar.scene_builder import (
    PlanarSceneBuilder,
)
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array, SceneConfig, SimConfig

MJCF_FILE = f"{os.path.join(os.path.dirname(__file__), 'assets/hopper.xml')}"

# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 0.6

# Hopping speed above which hop reward is 1.
_HOP_SPEED = 2


class HopperRobot(BaseAgent):
    uid = "hopper"
    mjcf_path = MJCF_FILE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.real_links = [
            link for link in self.robot.links_map.keys() if "dummy" not in link
        ]
        self.real_links_mass = torch.tensor(
            [
                link.mass[0].item()
                for link in self.robot.links
                if "dummy" not in link.name
            ],
            device=self.device,
        )
        self.real_mass = self.real_links_mass.sum()

    @property
    def _controller_configs(self):
        # NOTE joints in [rootx,rooty,rooz] are for planar tracking, not control joints
        # TODO (xhin): further tune controller params
        max_delta = 2.25
        stiffness = 100
        friction = 0.1
        force_limit = 200
        damping = 7.5
        pd_joint_delta_pos_body = PDJointPosControllerConfig(
            ["hip", "knee", "waist"],
            lower=-max_delta,
            upper=max_delta,
            damping=damping,
            stiffness=stiffness,
            force_limit=force_limit,
            friction=friction,
            use_delta=True,
        )
        pd_joint_delta_pos_ankle = PDJointPosControllerConfig(
            ["ankle"],
            lower=-max_delta / 3,
            upper=max_delta / 3,
            damping=damping,
            stiffness=stiffness,
            force_limit=force_limit,
            friction=friction,
            use_delta=True,
        )
        rest = PassiveControllerConfig(
            [j.name for j in self.robot.active_joints if "root" in j.name],
            damping=0,
            friction=0,
        )
        return deepcopy_dict(
            dict(
                pd_joint_delta_pos=dict(
                    body=pd_joint_delta_pos_body,
                    ankle=pd_joint_delta_pos_ankle,
                    rest=rest,
                    balance_passive_force=False,
                ),
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
        self.robot_link_ids = [link.name for link in self.robot.get_links()]

    # planar agent has root joints in range [-inf, inf], not ideal in obs space
    def get_proprioception(self):
        return dict(
            # don't include xslider qpos, for x trans invariance
            qpos=self.robot.get_qpos()[:, 1:],
            qvel=self.robot.get_qvel(),
        )


class HopperEnv(BaseEnv):
    agent: Union[HopperRobot]

    def __init__(self, *args, robot_uids=HopperRobot, **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            scene_cfg=SceneConfig(
                solver_position_iterations=4, solver_velocity_iterations=1
            ),
            sim_freq=100,
            control_freq=25,
        )

    @property
    def _default_sensor_configs(self):
        return [
            # replicated from xml file
            CameraConfig(
                uid="cam0",
                pose=sapien_utils.look_at(eye=[0, -2.8, 0.8], target=[0, 0, 0]),
                width=128,
                height=128,
                fov=45 * (np.pi / 180),
                near=0.01,
                far=100,
                mount=self.agent.robot.links_map["torso_dummy_1"],
            ),
        ]

    @property
    def _default_human_render_camera_configs(self):
        return [
            # replicated from xml file
            CameraConfig(
                uid="render_cam",
                pose=sapien_utils.look_at(eye=[0, -2.8, 0.8], target=[0, 0, 0]),
                width=512,
                height=512,
                fov=45 * (np.pi / 180),
                near=0.01,
                far=100,
                mount=self.agent.robot.links_map["torso_dummy_1"],
            ),
        ]

    def _load_scene(self, options: dict):
        loader = self.scene.create_mjcf_loader()
        articulation_builders, actor_builders, sensor_configs = loader.parse(MJCF_FILE)
        for a in actor_builders:
            a.build(a.name)

        self.planar_scene = PlanarSceneBuilder(env=self)
        self.planar_scene.build()

    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        with torch.device(self.device):
            self.planar_scene.initialize(env_idx)

    def evaluate(self):
        return dict()

    @property  # dm_control mjc function adapted for maniskill
    def height(self):
        """Returns relative height of the robot"""
        return (
            self.agent.robot.links_map["torso"].pose.p[:, -1]
            - self.agent.robot.links_map["foot_heel"].pose.p[:, -1]
        ).view(-1, 1)

    @property  # dm_control mjc function adapted for maniskill
    def subtreelinvelx(self):
        """Returns linvel x component of robot"""
        return self.agent.robot.get_qvel()[:, 0].view(-1, 1)

    # dm_control mjc function adapted for maniskill
    def touch(self, link_name):
        """Returns function of sensor force values"""
        force_vec = self.agent.robot.get_net_contact_forces([link_name])
        force_mag = torch.linalg.norm(force_vec, dim=-1)
        return torch.log1p(force_mag)

    # dm_control also includes foot pressures as state obs space
    def _get_obs_state_dict(self, info: Dict):
        return dict(
            agent=self._get_obs_agent(),
            toe_touch=self.touch("foot_toe"),
            heel_touch=self.touch("foot_heel"),
        )


@register_env("MS-HopperStand-v1", max_episode_steps=600)
class HopperStandEnv(HopperEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        return rewards.tolerance(self.height, lower=_STAND_HEIGHT, upper=2.0).view(-1)

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward


@register_env("MS-HopperHop-v1", max_episode_steps=600)
class HopperHopEnv(HopperEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        standing = rewards.tolerance(self.height, lower=_STAND_HEIGHT, upper=2.0)
        hopping = rewards.tolerance(
            self.subtreelinvelx,
            lower=_HOP_SPEED,
            upper=float("inf"),
            margin=_HOP_SPEED / 2,
            value_at_margin=0.5,
            sigmoid="linear",
        )
        return standing.view(-1) * hopping.view(-1)

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward