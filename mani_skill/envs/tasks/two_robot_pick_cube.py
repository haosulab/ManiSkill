from collections import OrderedDict
from typing import Any, Dict, Tuple

import numpy as np
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@register_env("TwoRobotPickCube-v1", max_episode_steps=100)
class TwoRobotPickCube(BaseEnv):
    """
    Task Description
    ----------------
    The goal is to pick up a red cube and lift it to a goal location. There are two robots in this task and the
    goal location is out of reach of the left robot while the cube is out of reach of the right robot, thus the two robots must work together
    to move the cube to the goal.

    Randomizations
    --------------
    - cube has its z-axis rotation randomized
    - cube has its xy positions on top of the table scene randomized such that it is in within reach of the left robot but not the right.
    - the target goal position (marked by a green sphere) of the cube is randomized such that it is within reach of the right robot but not the left.


    Success Conditions
    ------------------
    - red cube is at the goal location
    Visualization: TODO
    """

    SUPPORTED_ROBOTS = [("panda", "panda")]
    agent: MultiAgent[Tuple[Panda, Panda]]
    cube_half_size = 0.02
    goal_thresh = 0.025

    def __init__(
        self, *args, robot_uids=("panda", "panda"), robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_cfg(self):
        return SimConfig(
            gpu_memory_cfg=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**19,
                max_rigid_contact_count=2**21,
            )
        )

    @property
    def _sensor_configs(self):
        pose = sapien_utils.look_at([1.0, 0, 0.75], [0.0, 0.0, 0.25])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.4, 0.8, 0.75], [0.0, 0.1, 0.1])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self._scene,
            half_size=0.02,
            color=[1, 0, 0, 1],
            name="cube",
        )
        self.goal_site = actors.build_sphere(
            self._scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, 0] = torch.rand((b,)) * 0.1 - 0.05
            # ensure cube is spawned on the left side of the table
            xyz[:, 1] = -0.15 - torch.rand((b,)) * 0.1 + 0.05
            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, 0] = torch.rand((b,)) * 0.1 - 0.05
            goal_xyz[:, 1] = 0.15 + torch.rand((b,)) * 0.1 - 0.05
            goal_xyz[:, 2] = torch.rand((b,)) * 0.3 + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    @property
    def left_agent(self) -> Panda:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> Panda:
        return self.agent.agents[1]

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_right_arm_static = self.right_agent.is_static(0.2)
        return {
            "success": torch.logical_and(is_obj_placed, is_right_arm_static),
            "is_obj_placed": is_obj_placed,
            "is_right_arm_static": is_right_arm_static,
        }

    def _get_obs_extra(self, info: Dict):
        obs = OrderedDict(
            left_arm_tcp=self.left_agent.tcp.pose.raw_pose,
            right_arm_tcp=self.right_agent.tcp.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                cube_pose=self.cube.pose.raw_pose,
                left_arm_tcp_to_cube_pos=self.cube.pose.p - self.left_agent.tcp.pose.p,
                right_arm_tcp_to_cube_pos=self.cube.pose.p
                - self.right_agent.tcp.pose.p,
                cube_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Stage 1: Reach and push cube to be near other robot
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.left_agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)

        # set a sub_goal here where we want the cube to first be pushed to close to the right arm robot
        # by moving cube past y = 0.05
        cube_to_other_side_reward = 1 - torch.tanh(
            5
            * (
                torch.max(
                    0.05 - self.cube.pose.p[:, 1], torch.zeros_like(reaching_reward)
                )
            )
        )
        reward = (reaching_reward + cube_to_other_side_reward) / 2

        # stage 1 passes if cube is near a sub-goal
        cube_at_other_side = self.cube.pose.p[:, 1] >= 0.0

        # Stage 2: reach and grasp cube with right robot and make left robot leave space
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.right_agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        stage_2_reward = reaching_reward

        is_grasped = self.right_agent.is_grasping(self.cube)
        stage_2_reward += is_grasped

        # make left arm move as close as possible to the y=-0.2 line
        left_arm_leave_reward = 1 - torch.tanh(
            5 * (self.left_agent.tcp.pose.p[:, 1] + 0.2).abs()
        )
        stage_2_reward += left_arm_leave_reward

        reward[cube_at_other_side] = 2 + stage_2_reward[cube_at_other_side]

        # stage 2 passes if cube is grasped
        # is_grasped

        # Stage 3: place object at goal
        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward[is_grasped] = 6 + place_reward[is_grasped]

        # stage 3 passes if object is placed
        is_obj_placed = info["is_obj_placed"]

        # Stage 4: keep robot static at the goal
        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.right_agent.robot.get_qvel()[..., :-2], axis=1)
        )
        reward[is_obj_placed] = 8 + static_reward[is_obj_placed]

        reward[info["success"]] = 10
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 10
