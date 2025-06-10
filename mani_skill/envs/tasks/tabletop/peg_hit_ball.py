from typing import Any, Dict

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


@register_env("PegHitBall-v1", max_episode_steps=100)
class PegHitBallEnv(BaseEnv):
    """
    **Task Description:**
    Pick up a orange-white peg and strike a ball into a goal

    **Randomizations:**
    - The peg's xy position is randomized on top of the table in the region [-0.15, 0.15] x [0.40, 0.50]. Its half-length is sampled from [0.09, 0.12] and half-width/radius from [0.018, 0.025]; the peg lies flat with its head facing the ball/goal.
    - The ball's xy position is randomized on the table in the region [-0.15, -0.05]. The y-position is fixed. It is placed flat on the surface.
    - The goal region's xy position (red/white disk) is randomized on the table in the region [-0.15, 0.15] x [-0.65, -0.50].

    **Success Conditions:**
    - The ball's xy position is within goal_radius (default 0.1) of the target's xy position by euclidean distance.
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PegHitBall-v1_rt.mp4"

    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    goal_radius: float = 0.1
    ball_radius: float = 0.035

    def __init__(
        self,
        *args,
        robot_uids: str = "panda",
        robot_init_qpos_noise: float = 0.02,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**18,
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[-0.15, 0.85, 0.35], target=[0.0, 0.0, 0.0])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-0.6, 1.25, 0.8], [0.0, 0.0, 0.0])
        return CameraConfig("render_camera", pose, 512, 512, 1.0, 0.01, 100)

    def _load_agent(self, options: dict):
        """Spawn the Panda just behind the table."""
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0.60, 0.0]))

    def _load_scene(self, options: dict):
        """Create table, randomised peg, ball and goal – one set per env."""
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        rng = self._batched_episode_rng
        lengths_np = rng.uniform(0.09, 0.12)
        radii_np = rng.uniform(0.018, 0.025)

        self.peg_half_sizes = common.to_tensor(
            np.vstack([lengths_np, radii_np, radii_np]).T
        ).to(
            self.device
        )  # (N,3)

        pegs = []
        for env_i in range(self.num_envs):
            length = float(lengths_np[env_i])
            radius = float(radii_np[env_i])

            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=[length, radius, radius])

            head_mat = sapien.render.RenderMaterial(
                base_color=sapien_utils.hex2rgba("#EC7357"), roughness=0.5, specular=0.5
            )
            tail_mat = sapien.render.RenderMaterial(
                base_color=sapien_utils.hex2rgba("#EDF6F9"), roughness=0.5, specular=0.5
            )

            builder.add_box_visual(
                sapien.Pose([length, 0, 0]),
                half_size=[length, radius, radius],
                material=head_mat,
            )
            builder.add_box_visual(
                sapien.Pose([-length, 0, 0]),
                half_size=[length, radius, radius],
                material=tail_mat,
            )

            builder.initial_pose = sapien.Pose(p=[0, 0.3, 0.1])
            builder.set_scene_idxs([env_i])
            peg = builder.build(f"peg_{env_i}")
            self.remove_from_state_dict_registry(peg)  # merged later
            pegs.append(peg)

        self.peg = Actor.merge(pegs, "peg")
        self.add_to_state_dict_registry(self.peg)

        off_handle = torch.zeros((self.num_envs, 3))
        off_handle[:, 0] = -self.peg_half_sizes[:, 0]
        self.peg_handle_offsets = Pose.create_from_pq(p=off_handle)

        off_head = torch.zeros((self.num_envs, 3))
        off_head[:, 0] = self.peg_half_sizes[:, 0]
        self.peg_head_offsets = Pose.create_from_pq(p=off_head)

        self.ball = actors.build_sphere(
            self.scene,
            radius=self.ball_radius,
            color=[0.05, 0.2, 0.8, 1.0],
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

        self.ball_struck = torch.zeros(self.num_envs, dtype=torch.float32)

    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        """Randomise object poses and reset robot per given env_idx tensor."""
        self.ball_struck = self.ball_struck.to(self.device)

        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            robot_pose = Pose.create_from_pq(
                p=[-0.1, 1.0, 0], q=[0.7071, 0, 0, -0.7072]
            )
            self.agent.robot.set_pose(robot_pose)

            xyz_peg = torch.zeros((b, 3))
            xyz_peg[:, 0] = torch.rand(b) * 0.15 - 0.15
            xyz_peg[:, 1] = torch.rand(b) * 0.10 + 0.4
            xyz_peg[:, 2] = self.peg_half_sizes[env_idx, 2]
            quat_peg = torch.tensor(
                [0.70710678, 0.0, 0.0, -0.70710678], device=self.device
            ).repeat(b, 1)
            # quat_peg = randomization.random_quaternions(b, self.device, lock_x=True, lock_y=True)
            self.peg.set_pose(Pose.create_from_pq(p=xyz_peg, q=quat_peg))

            xyz_ball = torch.zeros((b, 3))
            xyz_ball[:, 0] = torch.rand(b) * 0.10 - 0.15
            # xyz_ball[:, 1] = torch.rand(b) * 0.15 + 0.25
            xyz_ball[:, 1] = 0.0  # fixed y in front of peg
            xyz_ball[:, 2] = self.ball_radius
            self.ball.set_pose(Pose.create_from_pq(p=xyz_ball, q=[1, 0, 0, 0]))

            xyz_goal = torch.zeros((b, 3))
            xyz_goal[:, 0] = torch.rand(b) * 0.3 - 0.15
            xyz_goal[:, 1] = -(torch.rand(b) * 0.15 + 0.5)
            xyz_goal[:, 2] = 1e-3
            self.goal_region.set_pose(
                Pose.create_from_pq(p=xyz_goal, q=euler2quat(0, np.pi / 2, 0))
            )

        self.ball_struck[env_idx] = 0.0

    def _get_obs_extra(self, info: Dict):
        obs = {"tcp_pose": self.agent.tcp.pose.raw_pose}
        if self.obs_mode_struct.use_state:
            obs.update(
                peg_pose=self.peg.pose.raw_pose,
                ball_pose=self.ball.pose.raw_pose,
                goal_pos=self.goal_region.pose.p,
                # tcp_to_peg_pos=self.peg.pose.p - self.agent.tcp.pose.p,
                # ball_to_goal_pos=self.goal_region.pose.p - self.ball.pose.p,
            )
        return obs

    def _peg_handle_pose(self):
        return self.peg.pose * self.peg_handle_offsets

    def _peg_head_pose(self):
        return self.peg.pose * self.peg_head_offsets

    def evaluate(self):
        ball_xy = self.ball.pose.p[..., :2]
        goal_xy = self.goal_region.pose.p[..., :2]
        success = torch.linalg.norm(ball_xy - goal_xy, dim=1) < self.goal_radius
        return {"success": success}

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        gripper_pos = self.agent.tcp.pose.p
        peg_handle_pos = self._peg_handle_pose().p
        d_gripper_handle = torch.linalg.norm(gripper_pos - peg_handle_pos, dim=1)
        reach_reward = 1 - torch.tanh(4.0 * d_gripper_handle)

        is_grasped = self.agent.is_grasping(self.peg, max_angle=20)
        grasp_bonus = is_grasped.to(torch.float32)

        peg_head_pos = self._peg_head_pose().p
        ball_pos = self.ball.pose.p
        d_head_ball = torch.linalg.norm(peg_head_pos - ball_pos, dim=1)
        strike_reward = (1 - torch.tanh(4.0 * d_head_ball)) * grasp_bonus

        # peg_bottom_z = self.peg.pose.p[:,2] - self.peg_half_sizes[:,2]
        # penetration = torch.clamp(0.003 - peg_bottom_z, min=0)
        # ground_penalty = 10 * penetration * is_grasped

        # Update struck flag if ball gains velocity
        ball_speed = torch.linalg.norm(self.ball.linear_velocity, dim=1)
        self.ball_struck[ball_speed > 0.2] = 1.0

        ball_xy = ball_pos[..., :2]
        goal_xy = self.goal_region.pose.p[..., :2]
        d_ball_goal = torch.linalg.norm(ball_xy - goal_xy, dim=1)
        goal_reward = (1 - torch.tanh(2.5 * d_ball_goal)) * self.ball_struck.to(
            ball_xy.dtype
        )

        reward = (
            reach_reward + grasp_bonus + strike_reward + 5 * goal_reward
        )  # - ground_penalty
        reward[info["success"]] = 20.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        return self.compute_dense_reward(obs, action, info) / 20.0
