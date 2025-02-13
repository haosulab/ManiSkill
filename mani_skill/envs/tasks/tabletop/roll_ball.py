from typing import Any, Dict

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


@register_env("RollBall-v1", max_episode_steps=80)
class RollBallEnv(BaseEnv):
    """
    **Task Description:**
    A simple task where the objective is to push and roll a ball to a goal region at the other end of the table

    **Randomizations:**
    - The ball's xy position is randomized on top of a table in the region [0.2, 0.5] x [-0.4, 0.7]. It is placed flat on the table
    - The target goal region is marked by a red/white circular target. The position of the target is randomized on top of a table in the region [-0.4, -0.7] x [0.2, -0.9]

    **Success Conditions:**
    - The ball's xy position is within goal_radius (default 0.1) of the target's xy position by euclidean distance.
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/RollBall-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda"]

    agent: Panda

    goal_radius: float = 0.1  # radius of the goal region
    ball_radius: float = 0.035  # radius of the ball
    reached_status: torch.Tensor

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
        pose = sapien_utils.look_at(eye=[-0.1, 0.9, 0.3], target=[0.0, 0.0, 0.0])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-0.6, 1.3, 0.8], [0.0, 0.13, 0.0])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.ball = actors.build_sphere(
            self.scene,
            radius=self.ball_radius,
            color=[0, 0.2, 0.8, 1],
            name="ball",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )

        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )
        self.reached_status = torch.zeros(self.num_envs, dtype=torch.float32)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.reached_status = self.reached_status.to(self.device)
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            robot_pose = Pose.create_from_pq(
                p=[-0.1, 1.0, 0], q=[0.7071, 0, 0, -0.7072]
            )
            self.agent.robot.set_pose(robot_pose)

            xyz = torch.zeros((b, 3))
            xyz[..., 0] = (torch.rand((b)) * 2 - 1) * 0.3 - 0.1
            xyz[..., 1] = torch.rand((b)) * 0.2 + 0.5
            xyz[..., 2] = self.ball_radius
            q = [1, 0, 0, 0]

            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.ball.set_pose(obj_pose)

            xyz_goal = torch.zeros((b, 3))
            xyz_goal[..., 0] = (torch.rand((b)) * 2 - 1) * 0.3 - 0.1
            xyz_goal[..., 1] = torch.rand((b)) * 0.2 - 1.0 + self.goal_radius
            xyz_goal[..., 2] = 1e-3
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=xyz_goal,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )
        self.reached_status[env_idx] = 0.0

    def evaluate(self):

        is_obj_placed = (
            torch.linalg.norm(
                self.ball.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
            )
            < self.goal_radius
        )

        return {
            "success": is_obj_placed,
        }

    def _get_obs_extra(self, info: Dict):

        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                goal_pos=self.goal_region.pose.p,
                ball_pose=self.ball.pose.raw_pose,
                ball_vel=self.ball.linear_velocity,
                tcp_to_ball_pos=self.ball.pose.p - self.agent.tcp.pose.p,
                ball_to_goal_pos=self.goal_region.pose.p - self.ball.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        unit_vec = self.ball.pose.p - self.goal_region.pose.p
        unit_vec = unit_vec / torch.linalg.norm(unit_vec, axis=1, keepdim=True)
        tcp_hit_pose = Pose.create_from_pq(
            p=self.ball.pose.p + unit_vec * (self.ball_radius + 0.05),
        )
        tcp_to_hit_pose = tcp_hit_pose.p - self.agent.tcp.pose.p
        tcp_to_hit_pose_dist = torch.linalg.norm(tcp_to_hit_pose, axis=1)
        self.reached_status[tcp_to_hit_pose_dist < 0.04] = 1.0
        reaching_reward = 1 - torch.tanh(2 * tcp_to_hit_pose_dist)

        obj_to_goal_dist = torch.linalg.norm(
            self.ball.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
        )

        reached_reward = 1 - torch.tanh(obj_to_goal_dist)

        reward = (
            20 * reached_reward * self.reached_status
            + reaching_reward * (1 - self.reached_status)
            + self.reached_status
        )

        reward[info["success"]] = 30.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        max_reward = 30.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
