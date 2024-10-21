from typing import Any, Dict, Union

import numpy as np
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


import sapien


@register_env("PullCubeTool-v1", max_episode_steps=100)
class PullCubeToolEnv(BaseEnv):
    """
    Task Description
    -----------------
    Given an L-shaped tool that is within the reach of the robot, leverage the
    tool to pull a cube that is out of it's reach


    Randomizations
    ---------------
    - The cube's position (x,y) is randomized on top of a table in the region "<out of manipulator
    reach, but within reach of tool>". It is placed flat on the table
    - The target goal region is the region on top of the table marked by "<within reach of arm>"

    Success Conditions
    -----------------
    - The cube's xy position is within the goal region of the arm's base (marked by reachability)
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    SUPPORTED_REWARD_MODES = ("normalized_dense", "dense", "sparse", "none")

    goal_radius = 0.1
    cube_half_size = 0.02

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        # defaulting to use panda arm
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

        self.handle_length = (0.15,)
        self.hook_length = (0.05,)
        self.width = (0.02,)
        self.height = 0.02

        self.cube_size = 0.04

        self.arm_reach = 0.85  # for setting boundary conditions of spawn

    # Specify default simulation/gpu memory configurations to override any default values
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        # to register a 128x128 camera looking at the robot, cube and target
        # set the camera's "eye"to be at 0.3,0,05 and the target pose as target
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.5], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        # registers a more high-definition (512x512) camera used just for rendering
        # when render_mode="rgb_array" or calling env.render_rgb_array()
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _build_l_shaped_tool(self, handle_length, hook_length, width, height):
        builder = self.scene.create_actor_builder()

        # Define material for the tool
        mat = sapien.render.RenderMaterial()
        mat.set_base_color([0.5, 0.5, 0.5, 1])
        mat.metallic = 0.0
        mat.roughness = 0.1

        # Add visual and collision shapes for the long part of the L
        builder.add_box_collision(
            sapien.Pose([handle_length / 2, 0, 0]),
            [handle_length / 2, width / 2, height / 2],
        )
        builder.add_box_visual(
            sapien.Pose([handle_length / 2, 0, 0]),
            [handle_length / 2, width / 2, height / 2],
            material=mat,
        )

        # Add visual and collision shapes for the short part of the L
        builder.add_box_collision(
            sapien.Pose([handle_length - hook_length / 2, width, 0]),
            [hook_length / 2, width / 2, height / 2],
        )
        builder.add_box_visual(
            sapien.Pose([handle_length - hook_length / 2, width, 0]),
            [hook_length / 2, width / 2, height / 2],
            material=mat,
        )

        return builder.build(name="l_shape_tool")

    def _load_scene(self, options: dict):
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()

        # Create the cube
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cube",
            body_type="dynamic",
        )

        # Create and position the L-shaped tool in the scene

        self.l_shape_tool = self._build_l_shaped_tool(
            handle_length=self.handle_length,
            hook_length=self.hook_length,
            width=self.width,
            height=self.height,
        )
        self.l_shape_tool.set_pose(
            sapien.Pose(p=[-0.1, -0.1, self.cube_half_size + self.height])
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)

            # Initialize robot position and orientation
            qpos = torch.tensor(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )

            qpos = (
                torch.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos)), device=self.device
                )
                + qpos
            )

            qpos[:, -2:] = 0.04
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

            # Initialize the tool

            tool_xyz = torch.zeros((b, 3))
            tool_xyz[..., :2] = (
                torch.rand((b, 2)) * 0.2 - 0.1
            )  # spawn tool in region where x,y in [-0.1, 0.1]
            tool_xyz[..., 2] = self.height / 2  # place tool on table
            tool_q = [1, 0, 0, 0]  # no rotation

            tool_pose = Pose.create_from_pq(p=tool_xyz, q=tool_q)
            self.l_shape_tool.set_pose(tool_pose)

            # Initialize the cube a bit away from the base of the arm

            cube_xyz = torch.zeros((b, 3))
            cube_xyz[..., 0] = self.arm_reach + torch.rand(b) * (
                self.handle_length - 0.08
            )  # Just outside arm's reach
            cube_xyz[..., 1] = torch.rand(b) * 0.4 - 0.2  # Random y position
            cube_xyz[..., 2] = self.cube_size / 2  # Place on the table

            cube_q = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
                bounds=(-np.pi / 6, np.pi / 6),
            )

            cube_pose = Pose.create_from_pq(p=cube_xyz, q=cube_q)
            self.cube.set_pose(cube_pose)

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            cube_pose=self.cube.pose.raw_pose,
            tool_pose=self.l_shape_tool.pose.raw_pose,
        )

        return obs

    def evaluate(self):

        tcp_pos = self.agent.tcp.pose.p
        cube_pos = self.cube.pose.p

        cube_in_reach = (
            torch.linalg.norm(tcp_pos[:, :2] - cube_pos[:, :2], dim=1)
            < self.goal_radius
        )
        cube_picked = self.agent.is_grasping(self.cube)

        return {
            "success": cube_picked,
            "cube_in_reach": cube_in_reach,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        
        tcp_pos = self.agent.tcp.pose.p
        cube_pos = self.cube.pose.p
        tool_pos = self.l_shape_tool.pose.p

        # Reward for reaching the tool
        tcp_to_tool_dist = torch.linalg.norm(tcp_pos - tool_pos, dim=1)
        reaching_tool_reward = 1 - torch.tanh(5.0 * tcp_to_tool_dist)

        # Reward for moving the tool towards the cube
        tool_to_cube_dist = torch.linalg.norm(tool_pos - cube_pos, dim=1)
        tool_cube_reward = 1 - torch.tanh(5.0 * tool_to_cube_dist)

        # Reward for bringing the cube closer to the robot
        tcp_to_cube_dist = torch.linalg.norm(tcp_pos - cube_pos, dim=1)
        cube_close_reward = 1 - torch.tanh(5.0 * tcp_to_cube_dist)

        # Success reward
        success_reward = info["success"].float() * 10

        # Combine rewards
        reward = (
            reaching_tool_reward + tool_cube_reward + cube_close_reward + success_reward
        )

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 13.0  # 10 + 1 + 1 + 1
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
