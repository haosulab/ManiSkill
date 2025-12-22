from typing import Any, Dict, Union
import os
import numpy as np
import sapien
import torch
from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@register_env("ScoopParticles-v1", max_episode_steps=100)
class ScoopParticlesEnv(BaseEnv):
    """
    **Task Description**
    Take a dustpan and scoop a ball onto it.

    **Randomizations**
    - The ball's (x,y) positions are randomized on top of a table.


    **Success Conditions**
    - The ball is inside the dustpan, lifted above the table height.
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PullCubeTool-v1_rt.mp4"

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    SUPPORTED_REWARD_MODES = ("normalized_dense", "dense", "sparse", "none")
    agent: Union[Panda, Fetch]

    goal_radius = 0.3
    ball_radius: float = 0.035  # radius of the ball
    handle_length = 0.2
    hook_length = 0.05
    width = 0.24
    height = 0.15
    cube_size = 0.02
    arm_reach = 0.35

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
        pose = sapien_utils.look_at([-0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return [
            CameraConfig(
                "render_camera",
                pose=pose,
                width=512,
                height=512,
                fov=1,
                near=0.01,
                far=100,
            )
        ]

    def _load_scene(self, options: dict):
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()

        self.ball = actors.build_sphere(
            self.scene,
            radius=ScoopParticlesEnv.ball_radius,
            color=[0, 0.2, 0.8, 1],
            name="ball",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )

        self.dustpan = self.load_glb_as_actor(
            self.scene,
            glb_file_path=os.path.join(PACKAGE_ASSET_DIR, 'scoop_particles/dustpan.glb'),
            pose=sapien.Pose(p=[0, 0, 0.015]),
            name="dustpan",
            type="dynamic"
        )

        self.wall = actors.build_box(self.scene, half_sizes=[0.5, 0.02, 0.2], color=
                                     [0.8, 0.3, 0.3, 1], name="wall", body_type="static", add_collision=True, initial_pose=sapien.Pose(p=[0.2, -0.25, 0.2], q=[0.7071, 0, 0, 0.7071]))

    @staticmethod
    def load_glb_as_actor(scene, glb_file_path, pose, name, type="static"):
        """Load GLB file as a static actor in the scene"""
        builder = scene.create_actor_builder()
        builder.add_visual_from_file(glb_file_path)
        builder.add_multiple_convex_collisions_from_file(glb_file_path, decomposition="coacd")
        builder.set_initial_pose(pose)
        if type=="dynamic":
            actor = builder.build_dynamic(name)
        else:
            actor = builder.build_static(name)
        return actor


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)

            tool_xyz = torch.zeros((b, 3), device=self.device)
            tool_xyz[..., :2] = -torch.rand((b, 2), device=self.device) * 0.2 - 0.1
            tool_xyz[..., 2] = 0.015
            tool_q = torch.tensor([0.559, 0.464, 0.439, 0.529], device=self.device).expand(b, 4)

            tool_pose = Pose.create_from_pq(p=tool_xyz, q=tool_q)
            self.dustpan.set_pose(tool_pose)

            ball_xyz = torch.zeros((b, 3), device=self.device)
            ball_xyz[..., 0] = (
                self.arm_reach
                + torch.rand(b, device=self.device) * (self.handle_length)
                - 0.3
            )
            ball_xyz[..., 1] = torch.rand(b, device=self.device) * 0.3 - 0.25
            ball_xyz[..., 2] = self.ball_radius + 0.01

            # ball_q = randomization.random_quaternions(
            #     b,
            #     lock_x=True,
            #     lock_y=True,
            #     lock_z=False,
            #     bounds=(-np.pi / 6, np.pi / 6),
            #     device=self.device,
            # )

            ball_pose = Pose.create_from_pq(p=ball_xyz, q=torch.tensor([1, 0, 0, 0], dtype=torch.float32))
            self.ball.set_pose(ball_pose)
            wall_xyz = torch.zeros((b, 3), device=self.device)
            wall_xyz[..., 0] = ball_xyz[..., 0] + 0.07
            wall_xyz[..., 1] = ball_xyz[..., 1]
            wall_xyz[..., 2] = 0.2
            self.wall.set_pose(Pose.create_from_pq(p=wall_xyz, q=torch.tensor([0.7071, 0, 0, 0.7071], dtype=torch.float32)))

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )

        if self.obs_mode_struct.use_state:
            obs.update(
                ball_pose=self.ball.pose.raw_pose,
                dustpan_pose=self.dustpan.pose.raw_pose,
            )

        return obs

    def evaluate(self):
        ball_pos = self.ball.pose.p

        dustpan_pos = self.dustpan.pose.p

        z_dist = ball_pos[..., 2] - dustpan_pos[..., 2]
        xy_dist = torch.linalg.norm(ball_pos[..., :2] - dustpan_pos[..., :2], dim=1)
        # Success condition - cube is pulled close enough
        ball_z_close_flag = torch.logical_and(z_dist < ScoopParticlesEnv.ball_radius + 0.02, z_dist > 0.0)
        ball_xy_close_flag = xy_dist < ScoopParticlesEnv.width * 1.414 / 2
        ball_pulled_close = torch.logical_and(ball_z_close_flag, ball_xy_close_flag)
        # is_ball_static = self.ball.is_static(lin_thresh=1e-1, ang_thresh=0.5)
        # print(is_ball_static)
        success = ball_pulled_close

        return {
            "success": success,
            "ball_pulled_close": ball_pulled_close,
            "ball_z_close_flag": ball_z_close_flag,
            "ball_xy_close_flag": ball_xy_close_flag,
            # "is_ball_static": is_ball_static,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):

        tcp_pos = self.agent.tcp.pose.p
        cube_pos = self.ball.pose.p
        tool_pos = self.dustpan.pose.p
        robot_base_pos = self.agent.robot.get_links()[0].pose.p

        # Stage 1: Reach and grasp tool
        tool_grasp_pos = tool_pos + torch.tensor([0.02, 0, 0], device=self.device)
        tcp_to_tool_dist = torch.linalg.norm(tcp_pos - tool_grasp_pos, dim=1)
        reaching_reward = 2.0 * (1 - torch.tanh(5.0 * tcp_to_tool_dist))

        # Add specific grasping reward
        is_grasping = self.agent.is_grasping(self.dustpan, max_angle=20)
        grasping_reward = 2.0 * is_grasping

        # Stage 2: Position tool behind cube
        ideal_hook_pos = cube_pos + torch.tensor(
            [-(self.hook_length + self.ball_radius), -0.067, 0], device=self.device
        )
        tool_positioning_dist = torch.linalg.norm(tool_pos - ideal_hook_pos, dim=1)
        positioning_reward = 1.5 * (1 - torch.tanh(3.0 * tool_positioning_dist))
        tool_positioned = tool_positioning_dist < 0.05

        # Stage 3: Pull cube to workspace
        workspace_target = robot_base_pos + torch.tensor(
            [0.05, 0, 0], device=self.device
        )
        cube_to_workspace_dist = torch.linalg.norm(cube_pos - workspace_target, dim=1)
        initial_dist = torch.linalg.norm(
            torch.tensor(
                [self.arm_reach + 0.1, 0, self.cube_size / 2], device=self.device
            )
            - workspace_target,
            dim=1,
        )
        pulling_progress = (initial_dist - cube_to_workspace_dist) / initial_dist
        pulling_reward = 3.0 * pulling_progress * tool_positioned

        # Combine rewards with staging and grasping dependency
        reward = reaching_reward + grasping_reward
        reward += positioning_reward * is_grasping
        reward += pulling_reward * is_grasping

        # Penalties
        cube_pushed_away = cube_pos[:, 0] > (self.arm_reach + 0.15)
        reward[cube_pushed_away] -= 2.0

        # Success bonus
        # if "success" in info:
        #     reward[info["success"]] += 5.0

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        """
        Normalizes the dense reward by the maximum possible reward (success bonus)
        """
        max_reward = 5.0  # Maximum possible reward from success bonus
        dense_reward = self.compute_dense_reward(obs=obs, action=action, info=info)
        return dense_reward / max_reward
