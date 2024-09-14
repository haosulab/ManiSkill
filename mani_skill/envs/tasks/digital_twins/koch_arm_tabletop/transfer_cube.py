from typing import Any, Dict

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.agents.robots.koch.koch import Koch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig


@register_env("TransferCube-v1", max_episode_steps=80)
class TransferCubeEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["koch-v1.1"]
    agent: Koch
    cube_half_size = 0.02
    goal_thresh = 0.025

    def __init__(
        self, *args, robot_uids="koch-v1.1", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien.Pose(
            [-0.00347404, 0.136826, 0.496307],
            [0.138651, 0.345061, 0.0516183, -0.926846],
        )
        return [CameraConfig("base_camera", pose, 640, 480, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien.Pose(
            [-0.00347404, 0.136826, 0.496307],
            [0.138651, 0.345061, 0.0516183, -0.926846],
        )
        return CameraConfig("render_camera", pose, 640, 480, 1, 0.01, 100)

    def build_square_boundary(
        self, size: float = 0.05, name: str = "square_boundary", color=[1, 1, 1, 1]
    ):
        builder = self.scene.create_actor_builder()
        border_thickness = 0.01  # Adjust this value as needed
        tape_thickness = 0.005
        # Top border
        builder.add_box_visual(
            pose=sapien.Pose([0, size - border_thickness / 2, 0]),
            half_size=[size, border_thickness / 2, tape_thickness / 2],
            material=sapien.render.RenderMaterial(base_color=color),
        )

        # Bottom border
        builder.add_box_visual(
            pose=sapien.Pose([0, -size + border_thickness / 2, 0]),
            half_size=[size, border_thickness / 2, tape_thickness / 2],
            material=sapien.render.RenderMaterial(base_color=color),
        )

        # Left border
        builder.add_box_visual(
            pose=sapien.Pose([-size + border_thickness / 2, 0, 0]),
            half_size=[border_thickness / 2, size, tape_thickness / 2],
            material=sapien.render.RenderMaterial(base_color=color),
        )

        # Right border
        builder.add_box_visual(
            pose=sapien.Pose([size - border_thickness / 2, 0, 0]),
            half_size=[border_thickness / 2, size, tape_thickness / 2],
            material=sapien.render.RenderMaterial(base_color=color),
        )
        return builder.build_kinematic(name)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene, half_size=self.cube_half_size, color=[1, 0, 0, 1], name="cube"
        )
        self.square_boundary_size = 0.1
        self.start_boundary = self.build_square_boundary(
            size=self.square_boundary_size, name="start_boundary"
        )
        self.end_boundary = self.build_square_boundary(
            size=self.square_boundary_size, name="goal_boundary"
        )

        self.rest_qpos = torch.from_numpy(self.agent.keyframes["rest"].qpos).to(
            self.device
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.Tensor([-0.4, -0.17])
            self.start_boundary.set_pose(Pose.create_from_pq(xyz))
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.Tensor([-0.4, 0.17])
            self.end_boundary.set_pose(Pose.create_from_pq(xyz))

            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                torch.rand((b, 2)) * self.square_boundary_size
                - self.square_boundary_size / 2
            )
            xyz[:, :2] += self.start_boundary.pose.p[env_idx, :2]
            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            # goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp.pose.p,
                obj_to_goal_pos=self.end_boundary.pose.p - self.cube.pose.p,
            )
        return obs

    def evaluate(self):
        within_end_boundary_xy = (
            self.cube.pose.p[:, :2]
            < self.end_boundary.pose.p[:, :2] + self.square_boundary_size
        ) & (
            self.cube.pose.p[:, :2]
            > self.end_boundary.pose.p[:, :2] - self.square_boundary_size
        )
        within_end_boundary_xy = within_end_boundary_xy.all(1)

        is_obj_placed = within_end_boundary_xy & (
            self.cube.pose.p[:, 2] < self.cube_half_size + 5e-3
        )
        robot_to_rest_pose_dist = torch.linalg.norm(
            self.agent.robot.qpos - self.rest_qpos, axis=1
        )
        is_grasped = self.agent.is_grasping(self.cube)
        return {
            "success": is_obj_placed & ~is_grasped & (robot_to_rest_pose_dist < 0.15),
            "is_obj_placed": is_obj_placed,
            "within_end_boundary_xy": within_end_boundary_xy,
            "robot_to_rest_pose_dist": robot_to_rest_pose_dist,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp.pose.p, axis=1
        )
        # stage 1, reach and grasp the object
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        # stage 2, move object over the goal boundary but encourage object is not too close to table
        obj_to_goal_dist = torch.linalg.norm(
            self.end_boundary.pose.p - self.cube.pose.p, axis=1
        )
        reach_goal_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        gripper_height_reward = torch.tanh(
            torch.clamp(self.agent.tcp.pose.p[:, 2], 0, 0.12) / 0.12
        ) / np.tanh(1 + 1e-3)
        reward[is_grasped] = (reach_goal_reward + gripper_height_reward + 3)[is_grasped]

        # stage 3, get object as close as possible to the middle of the goal boundary
        within_end_boundary_xy = info["within_end_boundary_xy"]
        reward[within_end_boundary_xy] = 5 + obj_to_goal_dist[within_end_boundary_xy]

        # stage 4, ensure object is at the desired location and not being grasped
        is_obj_placed = info["is_obj_placed"]
        ungrasp_reward = torch.tanh(
            torch.clamp(self.agent.robot.qpos[:, -1], -1, 0) / -1
        )
        reward[is_obj_placed] = (7 + ungrasp_reward)[is_obj_placed]

        # stage 5, return to rest qpos
        stage_5_cond = is_obj_placed & ~is_grasped
        reward[stage_5_cond] = (10 + (1 - torch.tanh(info["robot_to_rest_pose_dist"])))[
            stage_5_cond
        ]
        reward[info["success"]] = 12
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 12
