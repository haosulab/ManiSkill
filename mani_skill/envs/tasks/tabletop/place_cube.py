from typing import Any, Dict, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import sapien
import torch
import torch.random

from mani_skill.agents.robots import Fetch, Panda, XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


@register_env("PlaceCube-v1", max_episode_steps=150)
class PlaceCubeEnv(BaseEnv):
    """
    Task Description
    ----------------
    Place the cube into the bin.

    Randomizations
    --------------
    The position of the bin and the cube are randomized: The bin is inited in [0, 0.1]x[-0.1, 0.1], and the cube is inited in [-0.2, -0.15]x[-0.2, 0.2]

    Success Conditions
    ------------------
    The cube is placed inside the bin and static. The robot remains static and the gripper is not closed at the end state
    """

    SUPPORTED_ROBOTS = ["panda", "xarm6_robotiq"]

    # Specify some supported robot types
    agent: Union[Panda, XArm6Robotiq]

    # set some commonly used values
    cube_half_length = 0.02  # half side length of the cube
    inner_side_half_len = 0.06  # side length of the bin's inner square
    short_side_half_size = 0.008  # length of the shortest edge of the block
    block_half_size = [
        short_side_half_size,
        2 * short_side_half_size + inner_side_half_len,
        2 * short_side_half_size + inner_side_half_len,
    ]  # The bottom block of the bin, which is larger: The list represents the half length of the block along the [x, y, z] axis respectively.
    edge_block_half_size = [
        short_side_half_size,
        2 * short_side_half_size + inner_side_half_len,
        2 * short_side_half_size,
    ]  # The edge block of the bin, which is smaller. The representations are similar to the above one

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        real_pose = sapien_utils.look_at(eye=[0.508, -0.5, 0.42], target=[-0.522, 0.2, 0])
        return [
            # CameraConfig(
            #     "base_camera",
            #     pose=pose,
            #     width=128,
            #     height=128,
            #     fov=np.pi / 2,
            #     near=0.01,
            #     far=100,
            # ),
            CameraConfig(
                "real_camera",
                pose=real_pose,
                # width=640,
                # height=480,
                width=128,
                height=128,
                fov=np.pi * 0.4,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, -0.2, 0.2], [0.0, 0.0, 0.2])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _build_bin(self):
        builder = self.scene.create_actor_builder()

        dx = self.block_half_size[1] - self.block_half_size[0]
        dy = self.block_half_size[1] - self.block_half_size[0]
        dz = self.edge_block_half_size[2] + self.block_half_size[0]

        # build the bin bottom and edge blocks
        poses = [
            sapien.Pose([0, 0, 0]),
            sapien.Pose([-dx, 0, dz]),
            sapien.Pose([dx, 0, dz]),
            sapien.Pose([0, -dy, dz]),
            sapien.Pose([0, dy, dz]),
        ]
        half_sizes = [
            [self.block_half_size[1], self.block_half_size[2], self.block_half_size[0]],
            self.edge_block_half_size,
            self.edge_block_half_size,
            [
                self.edge_block_half_size[1],
                self.edge_block_half_size[0],
                self.edge_block_half_size[2],
            ],
            [
                self.edge_block_half_size[1],
                self.edge_block_half_size[0],
                self.edge_block_half_size[2],
            ],
        ]
        for pose, half_size in zip(poses, half_sizes):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size)

        builder.initial_pose = sapien.Pose(p=[0, 0, 0])
        return builder.build_kinematic(name="bin")

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise, custom_table=True
        )
        self.table_scene.build()

        self.obj = actors.build_cube(
            self.scene,
            half_size=self.cube_half_length,
            color=np.array([190, 30, 60, 255]) / 255,
            name="cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_length + 2*self.short_side_half_size]),
        )

        self.bin = self._build_bin()

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.5, 0, 0]))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # init the ~4cm cube in ([-0.2, -0.15]) along the x-axis (so that it doesn't collide with the bin)
            xyz = torch.zeros((b, 3))
            xyz[..., 0] = (torch.rand((b, 1)) * 0.05 - 0.2)[
                ..., 0
            ]
            # spanning ys from ([-0.2, 0.2])
            xyz[..., 1] = (torch.rand((b, 1)) * 0.4 - 0.2)[
                ..., 0
            ]
            xyz[..., 2] = self.cube_half_length  # on the table
            q = [1, 0, 0, 0]
            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.obj.set_pose(obj_pose)

            # init the ~12cm bin in ([0, 0.1]) along the x-axis (so that it doesn't collide with the cube)
            pos = torch.zeros((b, 3))
            pos[:, 0] = (
                torch.rand((b, 1))[..., 0] * 0.1
            )
            # spanning ys from ([-0.1, 0.1])
            pos[:, 1] = (
                torch.rand((b, 1))[..., 0] * 0.2 - 0.1
            )
            pos[:, 2] = self.block_half_size[0]  # on the table
            q = [1, 0, 0, 0]
            bin_pose = Pose.create_from_pq(p=pos, q=q)
            self.bin.set_pose(bin_pose)

    def evaluate(self):
        pos_obj = self.obj.pose.p
        pos_bin = self.bin.pose.p
        offset = pos_obj - pos_bin
        xy_flag = torch.linalg.norm(offset[..., :2], axis=1) <= 0.005
        entering_bin_z_flag = (
            offset[..., 2] - self.cube_half_length - self.short_side_half_size < 4 * self.short_side_half_size 
        )
        placed_in_bin_z_flag = (
            torch.abs(offset[..., 2] - self.cube_half_length - self.short_side_half_size) <= 0.005
        )
        is_obj_entering_bin = torch.logical_and(xy_flag, entering_bin_z_flag)
        is_obj_placed_in_bin = torch.logical_and(xy_flag, placed_in_bin_z_flag)
        is_obj_static = self.obj.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_obj_grasped = self.agent.is_grasping(self.obj)
        is_robot_static = self.agent.is_static(0.1)
        success = is_obj_placed_in_bin & is_obj_static & (~is_obj_grasped) & is_robot_static
        return {
            "is_obj_grasped": is_obj_grasped,
            "is_obj_entering_bin": is_obj_entering_bin,
            "is_obj_placed_in_bin": is_obj_placed_in_bin,
            "is_obj_static": is_obj_static,
            "is_robot_static": is_robot_static,
            "success": success,
        }

    def _get_obs_extra(self, info: Dict):
        # Only keep observables that are easy to get in the real world
        obs = dict(
            # is_grasped=info["is_obj_grasped"],
            # tcp_pose=self.agent.tcp.pose.raw_pose,
            bin_pos=self.bin.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                # obj_pose=self.obj.pose.raw_pose,
                tcp_to_obj_pos=self.obj.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward
        tcp_pose = self.agent.tcp.pose.p
        obj_pos = self.obj.pose.p
        obj_to_tcp_dist = torch.linalg.norm(tcp_pose - obj_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * obj_to_tcp_dist))

        # grasp and reach bin top reward
        obj_pos = self.obj.pose.p
        bin_top_pos = self.bin.pose.p.clone()
        # Height of the bin wall is 4 * short_side_half_size
        bin_top_pos[:, 2] = bin_top_pos[:, 2] + self.short_side_half_size + self.cube_half_length + 4 * self.short_side_half_size
        obj_to_bin_top_dist = torch.linalg.norm(bin_top_pos - obj_pos, axis=1)
        reach_bin_top_reward = 1 - torch.tanh(5.0 * obj_to_bin_top_dist)
        reward[info["is_obj_grasped"]] = (4 + reach_bin_top_reward)[info["is_obj_grasped"]]

        # reach inside bin reward
        bin_inside_pos = self.bin.pose.p.clone()
        bin_inside_pos[:, 2] = bin_inside_pos[:, 2] + self.short_side_half_size + self.cube_half_length + 0.01
        obj_to_bin_inside_dist = torch.linalg.norm(bin_inside_pos - obj_pos, axis=1)
        reach_bin_inside_reward = 1 - torch.tanh(5.0 * obj_to_bin_inside_dist)
        reward[info["is_obj_entering_bin"]] = (6 + reach_bin_inside_reward)[info["is_obj_entering_bin"]]

        # ungrasp and static reward
        is_obj_grasped = info["is_obj_grasped"]
        ungrasp_reward = self.agent.get_gripper_width()
        ungrasp_reward[
            ~is_obj_grasped
        ] = 1.0
        v = torch.linalg.norm(self.obj.linear_velocity, axis=1)
        av = torch.linalg.norm(self.obj.angular_velocity, axis=1)
        static_reward = 1 - torch.tanh(v * 5 + av)
        reward[info["is_obj_placed_in_bin"]] = (
            8 + (ungrasp_reward + static_reward) / 2.0
        )[info["is_obj_placed_in_bin"]]

        # go up and stay static reward
        robot_qvel = torch.linalg.norm(self.agent.robot.get_qvel(), axis=1)
        robot_static_reward = 1 - torch.tanh(5.0 * robot_qvel)
        tcp_to_bin_top_dist = torch.linalg.norm(self.agent.tcp.pose.p - self.bin.pose.p, axis=1)
        reach_bin_top_reward = 1 - torch.tanh(5.0 * tcp_to_bin_top_dist)
        
        final_state = info["is_obj_placed_in_bin"] & ~info["is_obj_grasped"]
        final_state_reward = (robot_static_reward + reach_bin_top_reward) / 2.0
        reward[final_state] = (10 + final_state_reward)[final_state]

        # success reward
        reward[info["success"]] = 15
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        max_reward = 15
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward