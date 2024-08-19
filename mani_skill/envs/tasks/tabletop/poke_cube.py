from typing import Any, Dict, Union

import numpy as np
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Fetch, Panda, Xmate3Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("PokeCube-v1", max_episode_steps=50)
class PokeCubeEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]
    agent: Union[Panda, Xmate3Robotiq, Fetch]

    cube_half_size = 0.02
    peg_half_width = 0.025
    peg_half_length = 0.12
    goal_radius = 0.05

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.2, 0.2, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            body_type="dynamic",
        )

        self.peg = actors.build_twocolor_peg(
            self.scene,
            length=self.peg_half_length,
            width=self.peg_half_width,
            color_1=np.array([12, 42, 160, 255]) / 255,
            color_2=np.array([12, 42, 160, 255]) / 255,
            name="peg",
            body_type="dynamic",
        )

        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
        )

        self.peg_head_offsets = Pose.create_from_pq(p=[self.peg_half_length, 0, 0])

    @property
    def peg_head_pos(self):
        return self.peg.pose.p + self.peg_head_offsets.p

    @property
    def peg_head_pose(self):
        return self.peg.pose * self.peg_head_offsets

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            # initialize the peg
            peg_xyz = torch.zeros((b, 3))
            peg_xyz = torch.rand((b, 3)) * 0.2 - 0.1
            peg_xyz[..., 2] = self.peg_half_width
            peg_q = [1, 0, 0, 0]
            peg_pose = Pose.create_from_pq(p=peg_xyz, q=peg_q)
            self.peg.set_pose(peg_pose)
            # initialize the cube
            cube_xyz = torch.zeros((b, 3))
            cube_xyz = torch.rand((b, 3)) * 0.2 - 0.1
            cube_xyz[..., 0] = peg_xyz[..., 0] + self.peg_half_length + 0.1
            cube_xyz[..., 2] = self.cube_half_size
            cube_q = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
                bounds=(-np.pi / 6, np.pi / 6),
            )
            cube_pose = Pose.create_from_pq(p=cube_xyz, q=cube_q)
            self.cube.set_pose(cube_pose)
            # initialize the goal region
            goal_region_xyz = cube_xyz + torch.tensor([0.05 + self.goal_radius, 0, 0])
            goal_region_xyz[..., 2] = 1e-3
            goal_region_q = euler2quat(0, np.pi / 2, 0)
            goal_region_pose = Pose.create_from_pq(p=goal_region_xyz, q=goal_region_q)
            self.goal_region.set_pose(goal_region_pose)

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )

        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                cube_pose=self.cube.pose.raw_pose,
                peg_pose=self.peg.pose.raw_pose,
                goal_pos=self.peg.pose.p,
                tcp_to_peg_pos=self.peg.pose.p - self.agent.tcp.pose.p,
                peg_to_cube_pos=self.cube.pose.p - self.peg.pose.p,
                cube_to_goal_pos=self.goal_region.pose.p - self.cube.pose.p,
                peghead_to_cube_pos=self.peg_head_pos - self.cube.pose.p,
            )
        return obs

    def evaluate(self):
        is_cube_placed = (
            torch.linalg.norm(
                self.cube.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
            )
            < self.goal_radius
        )
        peg_q = self.peg_head_pose.q
        peg_qmat = rotation_conversions.quaternion_to_matrix(peg_q)
        peg_euler = rotation_conversions.matrix_to_euler_angles(peg_qmat, "XYZ")
        cube_q = self.cube.pose.q
        cube_qmat = rotation_conversions.quaternion_to_matrix(cube_q)
        cube_euler = rotation_conversions.matrix_to_euler_angles(cube_qmat, "XYZ")
        angle_diff = torch.abs(peg_euler[:, 2] - cube_euler[:, 2])
        is_peg_cube_aligned = angle_diff < 0.05

        head_to_cube_dist = torch.linalg.norm(
            self.peg_head_pos[..., :2] - self.cube.pose.p[..., :2], axis=1
        )
        is_peg_cube_close = head_to_cube_dist <= self.cube_half_size + 0.005

        is_peg_cube_fit = torch.logical_and(is_peg_cube_aligned, is_peg_cube_close)
        is_peg_grasped = self.agent.is_grasping(self.peg)
        close_to_table = torch.abs(self.peg.pose.p[:, 2] - self.peg_half_width) < 0.005
        return {
            "success": is_cube_placed & is_peg_cube_fit & close_to_table,
            "is_peg_cube_fit": is_peg_cube_fit,
            "is_peg_grasped": is_peg_grasped,
            "angle_diff": angle_diff,
            "head_to_cube_dist": head_to_cube_dist,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reach peg
        tcp_pos = self.agent.tcp.pose.p
        tgt_tcp_pose = self.peg.pose
        tcp_to_peg_dist = torch.linalg.norm(tcp_pos - tgt_tcp_pose.p, axis=1)
        reached = tcp_to_peg_dist < 0.01
        reaching_reward = 2 * (1 - torch.tanh(5.0 * tcp_to_peg_dist))
        reward = reaching_reward

        # peg to cube
        angle_diff = info["angle_diff"]
        align_reward = 1 - torch.tanh(5.0 * angle_diff)
        head_to_cube_dist = info["head_to_cube_dist"]
        close_reward = 1 - torch.tanh(5.0 * head_to_cube_dist)
        is_peg_grasped = info["is_peg_grasped"] * reached
        reward[is_peg_grasped] = (4 + close_reward + align_reward)[is_peg_grasped]

        # cube to goal
        cube_to_goal_dist = torch.linalg.norm(
            self.goal_region.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * cube_to_goal_dist)
        is_peg_cube_fit = info["is_peg_cube_fit"] * is_peg_grasped
        reward[is_peg_cube_fit] = (7 + place_reward)[is_peg_cube_fit]

        reward[info["success"]] = 9
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 9.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
