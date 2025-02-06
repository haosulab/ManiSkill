from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.random

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import TriFingerPro
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils.randomization.pose import random_quaternions
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import ActorBuilder, actors
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Actor, Articulation, Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


class RotateCubeEnv(BaseEnv):
    """
    Modified from https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/trifinger.py
    https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/isaacgymenvs/cfg/task/Trifinger.yaml
    """

    SUPPORTED_ROBOTS = ["trifingerpro"]

    # Specify some supported robot types
    agent: TriFingerPro

    # Specify default simulation/gpu memory configurations.
    sim_config = SimConfig(
        gpu_memory_config=GPUMemoryConfig(
            found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
        )
    )

    # set some commonly used values
    goal_radius = 0.02
    cube_half_size = 0.02

    # radius of the area
    ARENA_RADIUS = 0.195
    size = 0.065  # m
    max_len = 0.065
    # 3D radius of the cuboid
    radius_3d = max_len * np.sqrt(3) / 2
    # compute distance from wall to the center
    max_com_distance_to_center = ARENA_RADIUS - radius_3d
    # minimum and maximum height for spawning the object
    min_height = 0.065 / 2
    max_height = 0.1

    def __init__(
        self,
        *args,
        robot_uids="trifingerpro",
        robot_init_qpos_noise=0.02,
        difficulty_level: int = 4,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise

        if (
            not isinstance(difficulty_level, int)
            or difficulty_level >= 5
            or difficulty_level < 0
        ):
            raise ValueError(
                f"Difficulty level must be a int within 0-4, but get {difficulty_level}"
            )

        self.difficulty_level = difficulty_level
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=(0.7, 0.0, 0.7), target=(0.0, 0.0, 0.0))
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=(0.7, 0.0, 0.7), target=(0.0, 0.0, 0.0))
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.ground = build_ground(self.scene, altitude=0)
        loader1 = self.scene.create_urdf_loader()
        loader1.fix_root_link = True
        loader1.name = "table"
        urdf_path = f"{PACKAGE_ASSET_DIR}/robots/trifinger/table_without_border.urdf"
        table: Articulation = loader1.load(urdf_path)

        builder: ActorBuilder = self.scene.create_actor_builder()
        high_table_boundary_file_name = f"{PACKAGE_ASSET_DIR}/robots/trifinger/robot_properties_fingers/meshes/high_table_boundary.stl"
        builder.add_nonconvex_collision_from_file(
            filename=high_table_boundary_file_name, scale=[1, 1, 1], material=None
        )
        builder.add_visual_from_file(filename=high_table_boundary_file_name)
        table_boundary: Actor = builder.build_static("table2")

        self.obj = actors.build_colorful_cube(
            self.scene,
            half_size=self.size / 2,
            color=np.array([169, 42, 12, 255]) / 255,
            name="cube",
            body_type="dynamic",
            add_collision=True,
        )

        self.obj_goal = actors.build_colorful_cube(
            self.scene,
            half_size=self.size / 2,
            color=np.array([12, 160, 42, 255]) / 255,
            name="cube_goal",
            body_type="kinematic",
            add_collision=False,
        )
        self._hidden_objects.append(self.obj_goal)

    def _initialize_actors(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[..., 2] = self.size / 2 + 0.005
            obj_pose = Pose.create_from_pq(p=xyz, q=[1, 0, 0, 0])
            self.obj.set_pose(obj_pose)
            pos, orn = self._sample_object_goal_poses(
                env_idx, difficulty=self.difficulty_level
            )
            self.obj_goal.set_pose(Pose.create_from_pq(p=pos, q=orn))
            self.prev_norms = None

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._initialize_actors(env_idx)
        self._initialize_agent(env_idx)

    def _sample_object_goal_poses(self, env_idx: torch.Tensor, difficulty: int):
        """Sample goal poses for the cube and sets them into the desired goal pose buffer.

        Args:
            instances: A tensor constraining indices of environment instances to reset.
            difficulty: Difficulty level. The higher, the more difficult is the goal.

        Possible levels are:
            - 0: Random goal position on the table, no orientation.
            - 1:  Random goal position on the table, including yaw orientation.
            - 2: Fixed goal position in the air with x,y = 0.  No orientation.
            - 3: Random goal position in the air, no orientation.
            - 4: Random goal pose in the air, including orientation.
        """
        b = len(env_idx)
        default_orn = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], dtype=torch.float, device=self.device
        ).repeat(b, 1)

        def random_xy() -> Tuple[torch.Tensor, torch.Tensor]:
            """Returns sampled uniform positions in circle (https://stackoverflow.com/a/50746409)"""
            # sample radius of circle
            radius = torch.sqrt(torch.rand(b, dtype=torch.float, device=self.device))
            radius *= self.max_com_distance_to_center
            # sample theta of point
            theta = 2 * np.pi * torch.rand(b, dtype=torch.float, device=self.device)
            # x,y-position of the cube
            x = radius * torch.cos(theta)
            y = radius * torch.sin(theta)

            return x, y

        def random_z(min_height: float, max_height: float) -> torch.Tensor:
            """Returns sampled height of the goal object."""
            z = torch.rand(b, dtype=torch.float, device=self.device)
            z = (max_height - min_height) * z + min_height
            return z

        if difficulty == 0:
            # Random goal position on the table, no orientation.
            pos_x, pos_y = random_xy()
            pos_z = self.size / 2
            orientation = default_orn
        elif difficulty == 1:
            # For initialization
            pos_x, pos_y = random_xy()
            pos_z = self.size / 2
            orientation = random_quaternions(
                b, lock_x=True, lock_y=True, device=self.device
            )
        elif difficulty == 2:
            # Fixed goal position in the air with x,y = 0.  No orientation.
            pos_x, pos_y = 0.0, 0.0
            pos_z = self.min_height + 0.05
            orientation = default_orn
        elif difficulty == 3:
            # Random goal position in the air, no orientation.
            pos_x, pos_y = random_xy()
            pos_z = random_z(min_height=self.min_height, max_height=self.max_height)
            orientation = default_orn
        elif difficulty == 4:
            # Random goal pose in the air, including orientation.
            # Note: Set minimum height such that the cube does not intersect with the
            #       ground in any orientation

            # pick x, y, z according to the maximum height / radius at the current point
            # in the cirriculum
            pos_x, pos_y = random_xy()
            pos_z = random_z(min_height=self.radius_3d, max_height=self.max_height)
            orientation = random_quaternions(b, device=self.device)
        else:
            msg = f"Invalid difficulty index for task: {difficulty}."
            raise ValueError(msg)

        pos_tensor = torch.zeros((b, 3), dtype=torch.float, device=self.device)
        pos_tensor[:, 0] = pos_x
        pos_tensor[:, 1] = pos_y
        pos_tensor[:, 2] = pos_z
        return pos_tensor, orientation

    def evaluate(self):
        obj_p = self.obj.pose.p
        goal_p = self.obj_goal.pose.p
        obj_q = self.obj.pose.q
        goal_q = self.obj_goal.pose.q

        is_obj_pos_close_to_goal = (
            torch.linalg.norm(obj_p - goal_p, axis=1) < self.goal_radius
        )

        is_obj_q_close_to_goal = common.quat_diff_rad(obj_q, goal_q) < 0.1

        is_success = is_obj_pos_close_to_goal & is_obj_q_close_to_goal

        return {
            "success": is_success,
        }

    def _initialize_agent(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)
            dof = self.agent.robot.dof
            if isinstance(dof, torch.Tensor):
                dof = dof[0]

            init_qpos = torch.zeros((b, dof))
            init_qpos += torch.randn((b, dof)) * self.robot_init_qpos_noise
            self.agent.reset(init_qpos)
            self.agent.robot.set_pose(
                Pose.create_from_pq(
                    torch.tensor([0.0, 0, self.size / 2 + 0.022]),
                    torch.tensor([1, 0, 0, 0]),
                )
            )

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            goal_pos=self.obj_goal.pose.p,
            goal_q=self.obj_goal.pose.q,
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                obj_p=self.obj.pose.p,
                obj_q=self.obj.pose.q,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        obj_pos = self.obj.pose.p
        obj_q = self.obj.pose.q
        goal_pos = self.obj_goal.pose.p
        goal_q = self.obj_goal.pose.q

        object_dist_weight = 5
        object_rot_weight = 5

        # Reward penalising finger movement

        tip_poses = self.agent.tip_poses
        # shape (N, 3 + 4, 3 fingers)

        finger_reach_object_dist_1 = torch.norm(
            tip_poses[:, :3, 0] - obj_pos, p=2, dim=-1
        )
        finger_reach_object_dist_2 = torch.norm(
            tip_poses[:, :3, 1] - obj_pos, p=2, dim=-1
        )
        finger_reach_object_dist_3 = torch.norm(
            tip_poses[:, :3, 2] - obj_pos, p=2, dim=-1
        )
        finger_reach_object_reward1 = 1 - torch.tanh(5 * finger_reach_object_dist_1)
        finger_reach_object_reward2 = 1 - torch.tanh(5 * finger_reach_object_dist_2)
        finger_reach_object_reward3 = 1 - torch.tanh(5 * finger_reach_object_dist_3)
        finger_reach_object_reward = (
            object_dist_weight
            * (
                finger_reach_object_reward1
                + finger_reach_object_reward2
                + finger_reach_object_reward3
            )
            / 3
        )

        # Reward for object distance
        object_dist = torch.norm(obj_pos - goal_pos, p=2, dim=-1)

        init_xyz_tensor = torch.tensor(
            [0, 0, 0.032], dtype=torch.float, device=self.device
        ).reshape(1, 3)
        init_z_dist = torch.norm(
            init_xyz_tensor
            - goal_pos[
                ...,
            ],
            p=2,
            dim=-1,
        )

        # object_dist_reward = object_dist_weight * dt * lgsk_kernel(object_dist, scale=50., eps=2.)

        object_dist_reward = 1 - torch.tanh(5 * object_dist)
        object_init_dist_reward = 1 - torch.tanh(5 * init_z_dist)
        object_dist_reward -= object_init_dist_reward

        init_z_tensor = torch.tensor(
            [0.032], dtype=torch.float, device=self.device
        ).reshape(1, 1)
        object_z_dist = torch.norm(obj_pos[..., 2:3] - goal_pos[..., 2:3], p=2, dim=-1)
        init_z_dist = torch.norm(init_z_tensor - goal_pos[..., 2:3], p=2, dim=-1)
        object_lift_reward = 5 * ((1 - torch.tanh(5 * object_z_dist)))
        object_init_z_reward = 5 * ((1 - torch.tanh(5 * init_z_dist)))

        object_lift_reward -= object_init_z_reward

        # extract quaternion orientation
        angles = common.quat_diff_rad(obj_q, goal_q)
        object_rot_reward = -1 * torch.abs(angles)
        pose_reward = (
            object_dist_weight * (object_dist_reward + object_lift_reward)
            + object_rot_weight * object_rot_reward
        )
        total_reward = finger_reach_object_reward + pose_reward
        total_reward = total_reward.clamp(-15, 15)
        total_reward[info["success"]] = 15
        return total_reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        self.max_reward = 15
        dense_reward = self.compute_dense_reward(obs=obs, action=action, info=info)
        norm_dense_reward = dense_reward / (2 * self.max_reward) + 0.5
        return norm_dense_reward


@register_env("TriFingerRotateCubeLevel0-v1", max_episode_steps=250)
class RotateCubeEnvLevel0(RotateCubeEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            robot_init_qpos_noise=0.02,
            difficulty_level=0,
            **kwargs,
        )


@register_env("TriFingerRotateCubeLevel1-v1", max_episode_steps=250)
class RotateCubeEnvLevel1(RotateCubeEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            robot_init_qpos_noise=0.02,
            difficulty_level=1,
            **kwargs,
        )


@register_env("TriFingerRotateCubeLevel2-v1", max_episode_steps=250)
class RotateCubeEnvLevel2(RotateCubeEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            robot_init_qpos_noise=0.02,
            difficulty_level=2,
            **kwargs,
        )


@register_env("TriFingerRotateCubeLevel3-v1", max_episode_steps=250)
class RotateCubeEnvLevel3(RotateCubeEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            robot_init_qpos_noise=0.02,
            difficulty_level=3,
            **kwargs,
        )


@register_env("TriFingerRotateCubeLevel4-v1", max_episode_steps=250)
class RotateCubeEnvLevel4(RotateCubeEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            robot_init_qpos_noise=0.02,
            difficulty_level=4,
            **kwargs,
        )
