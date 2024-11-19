from typing import Any, Dict, Tuple

import numpy as np
import sapien
import torch

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
    **Task Description:**
    The goal is to pick up a red cube and lift it to a goal location. There are two robots in this task and the
    goal location is out of reach of the left robot while the cube is out of reach of the right robot, thus the two robots must work together
    to move the cube to the goal.

    **Randomizations:**
    - cube has its z-axis rotation randomized
    - cube has its xy positions on top of the table scene randomized such that it is in within reach of the left robot but not the right.
    - the target goal position (marked by a green sphere) of the cube is randomized such that it is within reach of the right robot but not the left.


    **Success Conditions:**
    - red cube is at the goal location
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/refs/heads/main/figures/environment_demos/TwoRobotPickCube-v1_rt.mp4"

    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[Panda, Panda]]
    cube_half_size = 0.02
    goal_thresh = 0.025

    def __init__(
        self,
        *args,
        robot_uids=("panda_wristcam", "panda_wristcam"),
        robot_init_qpos_noise=0.02,
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**19,
                max_rigid_contact_count=2**21,
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([1.0, 0, 0.75], [0.0, 0.0, 0.25])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.4, 0.8, 0.75], [0.0, 0.1, 0.1])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(
            options, [sapien.Pose(p=[0, -1, 0]), sapien.Pose(p=[0, 1, 0])]
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, 0.02]),
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.left_init_qpos = self.left_agent.robot.get_qpos()
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
        obs = dict(
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

        # condition for good grasp: both fingers are at the same height and open
        self.right_agent: Panda
        right_tip_1_height = self.right_agent.finger1_link.pose.p[:, 2]
        right_tip_2_height = self.right_agent.finger2_link.pose.p[:, 2]
        tip_height_reward = 1 - torch.tanh(
            5 * torch.abs(right_tip_1_height - right_tip_2_height)
        )
        tip_width_reward = 1 - torch.tanh(
            5
            * torch.abs(
                torch.linalg.norm(
                    self.right_agent.finger1_link.pose.p
                    - self.right_agent.finger2_link.pose.p,
                    axis=1,
                )
                - 0.07
            )
        )
        tip_reward = (tip_height_reward + tip_width_reward) / 2
        stage_2_reward += tip_reward

        # make left arm move as close as possible to the y=-0.2 line
        left_arm_leave_reward = 1 - torch.tanh(
            5 * (self.left_agent.tcp.pose.p[:, 1] + 0.2).abs()
        )
        stage_2_reward += left_arm_leave_reward

        # stage 2 passes if cube is grasped
        is_grasped = self.right_agent.is_grasping(self.cube)
        stage_2_reward += 2 * is_grasped

        reward[cube_at_other_side] = 2 + stage_2_reward[cube_at_other_side]

        # Stage 3: bring cube towards goal
        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.right_agent.tcp.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        stage_3_reward = 2 * place_reward

        # return left arm to original position
        left_qpos_reward = 1 - torch.tanh(
            torch.linalg.norm(
                self.left_agent.robot.get_qpos() - self.left_init_qpos, axis=1
            )
        )
        stage_3_reward += left_qpos_reward

        reward[is_grasped] = 8 + stage_3_reward[is_grasped]

        # stage 3 passes if object is near goal (within 0.25m) - intermediate reward
        is_obj_near = torch.logical_and(obj_to_goal_dist < 0.25, is_grasped)
        # Stage 4: reuse same reward as stage 3 but stronger incentive
        reward[is_obj_near] = 12 + 2 * stage_3_reward[is_obj_near]

        # stage 4 passes if object is placed
        is_obj_placed = info["is_obj_placed"]

        # Stage 5: keep robot static at the goal
        right_static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.right_agent.robot.get_qvel()[..., :-2], axis=1)
        )
        left_static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.left_agent.robot.get_qvel()[..., :-2], axis=1)
        )
        static_reward = (right_static_reward + left_static_reward) / 2

        reward[is_obj_placed] = 19 + static_reward[is_obj_placed]

        reward[info["success"]] = 21

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 21
