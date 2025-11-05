"""Adapted from https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/hopper.py"""

import os
from typing import Any, Optional, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization, rewards
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.geometry import rotation_conversions
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

    @property
    def _controller_configs(self):
        # NOTE joints in [rootx,rooty,rooz] are for planar tracking, not control joints
        max_delta = 2  # best by far
        stiffness = 100
        damping = 10  # end best
        pd_joint_delta_pos_body = PDJointPosControllerConfig(
            ["hip", "knee", "waist"],
            lower=-max_delta,
            upper=max_delta,
            damping=damping,
            stiffness=stiffness,
            use_delta=True,
        )
        pd_joint_delta_pos_ankle = PDJointPosControllerConfig(
            ["ankle"],
            lower=-max_delta / 2.5,
            upper=max_delta / 2.5,
            damping=damping,
            stiffness=stiffness,
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

    def _load_articulation(
        self, initial_pose: Optional[Union[sapien.Pose, Pose]] = None
    ):
        """
        Load the robot articulation
        """
        loader = self.scene.create_mjcf_loader()
        asset_path = str(self.mjcf_path)

        loader.name = self.uid

        builder = loader.parse(asset_path)["articulation_builders"][0]
        builder.initial_pose = initial_pose
        self.robot = builder.build()
        assert self.robot is not None, f"Fail to load URDF/MJCF from {asset_path}"
        self.robot_link_ids = [link.name for link in self.robot.get_links()]

        # cache robot mass for com computation
        self.robot_links_mass = [link.mass[0].item() for link in self.robot.get_links()]
        self.robot_mass = np.sum(self.robot_links_mass[3:])

    # planar agent has root joints in range [-inf, inf], not ideal in obs space
    def get_proprioception(self):
        return dict(
            # don't include xslider qpos, for x trans invariance
            # x increases throughout successful episode
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
            scene_config=SceneConfig(
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
                fov=np.pi / 4,
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
                fov=np.pi / 4,
                near=0.01,
                far=100,
                mount=self.agent.robot.links_map["torso_dummy_1"],
            ),
        ]

    def _load_scene(self, options: dict):
        loader = self.scene.create_mjcf_loader()
        actor_builders = loader.parse(MJCF_FILE)["actor_builders"]
        for a in actor_builders:
            a.build(a.name)

        self.planar_scene = PlanarSceneBuilder(env=self)
        self.planar_scene.build()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            # qpos sampled same as dm_control, but ensure no self intersection explicitly here
            random_qpos = torch.rand(b, self.agent.robot.dof[0])
            q_lims = self.agent.robot.get_qlimits()
            q_ranges = q_lims[..., 1] - q_lims[..., 0]
            random_qpos *= q_ranges
            random_qpos += q_lims[..., 0]

            # overwrite planar joint qpos - these are special for planar robots
            # first two joints are dummy rootx and rootz
            random_qpos[:, :2] = 0
            # y is axis of rotation of our planar robot (xz plane), so we randomize around it
            random_qpos[:, 2] = torch.pi * (2 * torch.rand(b) - 1)  # (-pi,pi)
            self.agent.reset(random_qpos)

    @property  # dm_control mjc function adapted for maniskill
    def height(self):
        """Returns relative height of the robot"""
        return (
            self.agent.robot.links_map["torso"].pose.p[:, -1]
            - self.agent.robot.links_map["foot_heel"].pose.p[:, -1]
        ).view(-1, 1)

    @property  # dm_control mjc function adapted for maniskill
    def subtreelinvelx(self):
        # """Returns linvel x component of robot"""
        links = self.agent.robot.get_links()[3:]  # skip first three dummy links
        vels = torch.stack(
            [link.get_linear_velocity() * link.mass[0].item() for link in links], dim=0
        )  # (num_links, b, 3)
        com_vel = vels.sum(dim=0) / self.agent.robot_mass  # (b, 3)
        return com_vel[:, 0]

    # dm_control mjc function adapted for maniskill
    def touch(self, link_name):
        """Returns function of sensor force values"""
        force_vec = self.agent.robot.get_net_contact_forces([link_name])
        force_mag = torch.linalg.norm(force_vec, dim=-1)
        return torch.log1p(force_mag)

    # dm_control also includes foot pressures as state obs space
    def _get_obs_state_dict(self, info: dict):
        return dict(
            agent=self._get_obs_agent(),
            toe_touch=self.touch("foot_toe"),
            heel_touch=self.touch("foot_heel"),
        )


@register_env("MS-HopperStand-v1", max_episode_steps=600)
class HopperStandEnv(HopperEnv):
    """
    **Task Description:**
    Hopper robot stands upright

    **Randomizations:**
    - Hopper robot is randomly rotated [-pi, pi] radians about y axis.
    - Hopper qpos are uniformly sampled within their allowed ranges

    **Success Conditions:**
    - No specific success conditions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        standing = rewards.tolerance(self.height, lower=_STAND_HEIGHT, upper=2.0)
        return standing.view(-1)

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward


@register_env("MS-HopperHop-v1", max_episode_steps=600)
class HopperHopEnv(HopperEnv):
    """
    **Task Description:**
    Hopper robot stays upright and moves in positive x direction with hopping motion

    **Randomizations:**
    - Hopper robot is randomly rotated [-pi, pi] radians about y axis.
    - Hopper qpos are uniformly sampled within their allowed ranges

    **Success Conditions:**
    - No specific success conditions. The task is considered successful if the hopper hops for the whole episode.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
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

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
