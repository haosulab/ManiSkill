"""
Code for a minimal environment/task with just a robot being loaded. We recommend copying this template and modifying as you need.

At a high-level, ManiSkill tasks can minimally be defined by how the environment resets, what agents/objects are
loaded, goal parameterization, and success conditions

Environment reset is comprised of running two functions, `self._reconfigure` and `self.initialize_episode`, which is auto
run by ManiSkill. As a user, you can override a number of functions that affect reconfiguration and episode initialization.

Reconfiguration will reset the entire environment scene and allow you to load/swap assets and agents.

Episode initialization will reset the positions of all objects (called actors), articulations, and agents,
in addition to initializing any task relevant data like a goal

See comments for how to make your own environment and what each required function should do
"""

from typing import Any, Dict, Union

import numpy as np
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Fetch, Panda, Xmate3Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


@register_env("GroupCube-v1", max_episode_steps=200)
class GroupCubeEnv_v1(BaseEnv):
    """
    Task Description
    ----------------
    A simple task where the objective is to push and move a cube to a goal region in front of it

    Randomizations
    --------------
    - the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
    - the target goal region is marked by a red/white circular target. The position of the target is fixed to be the cube xy position + [0.1 + goal_radius, 0]

    Success Conditions
    ------------------
    - the cube's xy position is within goal_radius (default 0.1) of the target's xy position by euclidean distance.

    Visualization: https://maniskill.readthedocs.io/en/latest/tasks/index.html#pushcube-v1
    """

    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]

    # Specify some supported robot types
    agent: Union[Panda, Xmate3Robotiq, Fetch]

    # set some commonly used values
    goal_radius = 0.1
    cube_half_size = 0.02

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        # specifying robot_uids="panda" as the default means gym.make("PushCube-v1") will default to using the panda arm.
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # Specify default simulation/gpu memory configurations to override any default values
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_cfg=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        # registers one 128x128 camera looking at the robot, cube, and target
        # a smaller sized camera will be lower quality, but render faster
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
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
        # registers a more high-definition (512x512) camera used just for rendering when render_mode="rgb_array" or calling env.render_rgb_array()
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_scene(self, options: dict):
        # we use a prebuilt scene builder class that automatically loads in a floor and table.
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # we then add the cube that we want to push and give it a color and size using a convenience build_cube function
        # we specify the body_type to be "dynamic" as it should be able to move when touched by other objects / the robot
        self.cubeA = actors.build_cube(
            self.scene, half_size=self.cube_half_size, color=[1, 0, 0, 1], name="cubeA"
        )
        self.cubeB = actors.build_cube(
            self.scene, half_size=self.cube_half_size, color=[0, 1, 0, 1], name="cubeB"
        )
        self.cubeC = actors.build_cube(
            self.scene, half_size=self.cube_half_size, color=[0, 0, 1, 1], name="cubeC"
        )
        self.cubeD = actors.build_cube(
            self.scene, half_size=self.cube_half_size, color=[1, 0, 1, 1], name="cubeD"
        )

        # we also add in red/white target to visualize where we want the cube to be pushed to
        # we specify add_collisions=False as we only use this as a visual for videos and do not want it to affect the actual physics
        # we finally specify the body_type to be "kinematic" so that the object stays in place
        self.goal_region1 = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region1",
            add_collision=False,
            body_type="kinematic",
        )
        self.goal_region2 = actors.build_red_black_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region2",
            add_collision=False,
            body_type="kinematic",
        )

        # optionally you can automatically hide some Actors from view by appending to the self._hidden_objects list. When visual observations
        # are generated or env.render_sensors() is called or env.render() is called with render_mode="sensors", the actor will not show up.
        # This is useful if you intend to add some visual goal sites as e.g. done in PickCube that aren't actually part of the task
        # and are there just for generating evaluation videos.
        # self._hidden_objects.append(self.goal_region)
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
    # use the torch.device context manager to automatically create tensors on CPU or CUDA depending on self.device, the device the environment runs on
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Randomize positions for the cubes
            xyz = torch.zeros((b, 4, 3))  # 4 cubes
            xyz[..., :2] = torch.rand((b, 4, 2)) * 0.2 - 0.1
            xyz[..., 2] = self.cube_half_size
            q = [1, 0, 0, 0]
            
            # Initialize poses for the cubes
            for i, cube in enumerate([self.cubeA, self.cubeB, self.cubeC, self.cubeD]):
                obj_pose = Pose.create_from_pq(p=xyz[:, i, :], q=q)
                cube.set_pose(obj_pose)

            # Set the locations for the goal regions
            target_region_xyz1 = xyz[:, 0, :] + torch.tensor([0.1 + self.goal_radius, 0.2, 0])
            target_region_xyz2 = xyz[:, 0, :] + torch.tensor([0.1 + self.goal_radius, -0.2, 0])
            
            target_region_xyz1[..., 2] = 1e-3
            target_region_xyz2[..., 2] = 1e-3
            
            self.goal_region1.set_pose(
                Pose.create_from_pq(
                    p=target_region_xyz1,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )
            self.goal_region2.set_pose(
                Pose.create_from_pq(
                    p=target_region_xyz2,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )
    def evaluate(self):
        # success is achieved when the cube's xy position on the table is within the
        # goal region's area (a circle centered at the goal region's xy position)
        is_cubeA_placed = (
            torch.linalg.norm(
                self.cubeA.pose.p[..., :2] - self.goal_region1.pose.p[..., :2], axis=1
            )
            < self.goal_radius
        )
        is_cubeB_placed = (
            torch.linalg.norm(
                self.cubeB.pose.p[..., :2] - self.goal_region1.pose.p[..., :2], axis=1
            )
            < self.goal_radius
        )
        is_cubeC_placed = (
            torch.linalg.norm(
                self.cubeC.pose.p[..., :2] - self.goal_region2.pose.p[..., :2], axis=1
            )
            < self.goal_radius
        )
        is_cubeD_placed = (
            torch.linalg.norm(
                self.cubeD.pose.p[..., :2] - self.goal_region2.pose.p[..., :2], axis=1
            )
            < self.goal_radius
        )
        success= is_cubeA_placed*is_cubeB_placed*is_cubeC_placed*is_cubeD_placed
        return {
            "success":success.bool(),
            "is_cubeA_placed": is_cubeA_placed,
            "is_cubeB_placed": is_cubeB_placed,
            "is_cubeC_placed": is_cubeC_placed,
            "is_cubeD_placed": is_cubeD_placed,
        }

    def _get_obs_extra(self, info: Dict):
        # some useful observation info for solving the task includes the pose of the tcp (tool center point) which is the point between the
        # grippers of the robot
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self._obs_mode in ["state", "state_dict"]:
            # if the observation mode is state/state_dict, we provide ground truth information about where the cubes are.
            # for visual observation modes one should rely on the sensed visual data to determine where the cubes are
            obs.update(
                goal_pos1=self.goal_region1.pose.p,
                goal_pos2=self.goal_region2.pose.p,
                obj_poses={
                    "cubeA": self.cubeA.pose.raw_pose,
                    "cubeB": self.cubeB.pose.raw_pose,
                    "cubeC": self.cubeC.pose.raw_pose,
                    "cubeD": self.cubeD.pose.raw_pose,
                }
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        # We also create a pose marking where the robot should push the cube from that is easiest (pushing from behind the cube)
        def _compute_reward(cube, goal,success):
            tcp_push_pose = Pose.create_from_pq(
                p=cube.pose.p
                + torch.tensor([-self.cube_half_size - 0.005, 0, 0], device=self.device)
            )
            tcp_to_push_pose = tcp_push_pose.p - self.agent.tcp.pose.p
            tcp_to_push_pose_dist = torch.linalg.norm(tcp_to_push_pose, axis=1)
            reaching_reward = 1 - torch.tanh(5 * tcp_to_push_pose_dist)
            reward = reaching_reward

            # compute a placement reward to encourage robot to move the cube to the center of the goal region
            # we further multiply the place_reward by a mask reached so we only add the place reward if the robot has reached the desired push pose
            # This reward design helps train RL agents faster by staging the reward out.
            reached = tcp_to_push_pose_dist < 0.01
            obj_to_goal_dist = torch.linalg.norm(
                cube.pose.p[..., :2] - goal.pose.p[..., :2], axis=1
            )
            place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
            reward += place_reward * reached

            # assign rewards to parallel environments that achieved success to the maximum of 3.
            reward[info[success]] = 3
            return reward
        reward_A = _compute_reward(self.cubeA, self.goal_region1,"is_cubeA_placed")
        reward_B = _compute_reward(self.cubeB, self.goal_region1,"is_cubeB_placed")
        reward_C = _compute_reward(self.cubeC, self.goal_region2,"is_cubeC_placed")
        reward_D = _compute_reward(self.cubeD, self.goal_region2,"is_cubeD_placed")
        reward=reward_A+2*reward_B*info["is_cubeA_placed"]+ \
        3*reward_C*info["is_cubeA_placed"]*info["is_cubeB_placed"]+ \
        4*reward_D*info["is_cubeA_placed"]*info["is_cubeB_placed"]*info["is_cubeC_placed"]
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 30.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
