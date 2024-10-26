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
from mani_skill.utils.structs import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig


# grab cube and return to rest keyframe
@register_env("GrabCube-v1", max_episode_steps=50)
class GrabCubeEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["koch-v1.1"]
    agent: Koch
    # spawn box params
    spawn_box_half_size = 0.175
    resting_tolerance = 0.1
    # Domain Randomization params
    # uniform dist to encourage robust policy
    # TODO: (xhin) test normal dist in future
    # differing cube sizes
    min_cube_half_size = 0.015
    max_cube_half_size = 0.02
    # min_cube_half_size = 0.0175
    # max_cube_half_size = 0.0175

    # min_cube_friction = 0.1
    # max_cube_friction = 2
    min_cube_friction = 0.3
    max_cube_friction = 0.3

    def __init__(
        self,
        *args,
        robot_uids="koch-v1.1",
        robot_init_qpos_noise=0.02,
        reconfiguration_freq=1,
        num_envs=1,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        # hard set for now while testing environment
        reconfiguration_freq = 1
        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien.Pose(
            [-0.00347404, 0.136826, 0.496307],
            [0.138651, 0.345061, 0.0516183, -0.926846],
        )
        return [CameraConfig("base_camera", pose, 640, 480, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0.4, 0.3], target=[-0.2, 0, 0.00])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

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

        self.spawn_box = self.build_square_boundary(
            size=self.spawn_box_half_size, name="spawn_boundary"
        )

        self.rest_qpos = torch.from_numpy(self.agent.keyframes["rest"].qpos).to(
            self.device
        )

        self.table_scene.build()
        cubes = []
        half_sizes = []
        # CUBE DR - size and friction
        # TODO (xhin) - cube color DR
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            half_size = (
                torch.rand(1).item()
                * (self.max_cube_half_size - self.min_cube_half_size)
                + self.min_cube_half_size
            )
            half_sizes.append(half_size)
            friction = (
                torch.rand(1).item() * (self.max_cube_friction - self.min_cube_friction)
                + self.min_cube_friction
            )
            material = sapien.pysapien.physx.PhysxMaterial(
                static_friction=friction,
                dynamic_friction=friction,
                restitution=0,
            )
            builder.add_box_collision(half_size=[half_size] * 3, material=material)
            builder.add_box_visual(
                half_size=[half_size] * 3,
                material=sapien.render.RenderMaterial(
                    base_color=[1, 0, 0, 1],
                ),
            )
            builder.set_scene_idxs([i])
            cube = builder.build(f"cube_{i}")
            cubes.append(cube)
        self.cube = Actor.merge(cubes, "cube")
        self.cube_half_sizes = torch.tensor(half_sizes).to(self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 0] = -0.175
            self.spawn_box.set_pose(Pose.create_from_pq(p=xyz))

            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                torch.rand((b, 2)) * self.spawn_box_half_size * 2
                - self.spawn_box_half_size
            )
            xyz[:, :2] += self.spawn_box.pose.p[env_idx, :2]
            xyz[:, 2] = self.cube_half_sizes[env_idx]
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict()
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p
                - (self.agent.tcp.pose.p + self.agent.tcp2.pose.p) / 2,
                is_grasped=info["is_grasped"],
                tcp_pose=self.agent.tcp.pose.raw_pose,
                tcp2_pose=self.agent.tcp2.pose.raw_pose,
                cube_side_length=self.cube_half_sizes * 2,
                grippers_distance=torch.linalg.norm(
                    self.agent.tcp.pose.p - self.agent.tcp2.pose.p, axis=-1
                ),
            )
        return obs

    def evaluate(self):
        is_grasped = self.agent.is_grasping(self.cube)
        # don't include the grasping in requirement of end qpos
        robot_to_rest_pose_dist = torch.linalg.norm(
            self.agent.robot.qpos[..., :-1] - self.rest_qpos[:-1], axis=1
        )
        robot_to_rest_pose_dist < self.resting_tolerance
        return {
            # "success": is_grasped & grapsed_and_resting,
            "is_grasped": is_grasped,
            "robot_to_grasped_rest_dist": robot_to_rest_pose_dist,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - (self.agent.tcp.pose.p + self.agent.tcp2.pose.p) / 2,
            axis=1,
        )
        # stage 1, reach tcp to object
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        # stage 2, grasp the object
        is_grasped = info["is_grasped"]
        reward += is_grasped

        # stage 2, return to rest position while grasping
        reward[is_grasped] += (
            1 - torch.tanh(info["robot_to_grasped_rest_dist"])[is_grasped]
        )
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 3


# from typing import Any, Dict

# import numpy as np
# import sapien
# import torch

# import mani_skill.envs.utils.randomization as randomization
# from mani_skill.agents.robots import Fetch, Panda
# from mani_skill.agents.robots.koch.koch import Koch
# from mani_skill.envs.sapien_env import BaseEnv
# from mani_skill.sensors.camera import CameraConfig
# from mani_skill.utils import sapien_utils
# from mani_skill.utils.building import actors
# from mani_skill.utils.registration import register_env
# from mani_skill.utils.scene_builder.table import TableSceneBuilder
# from mani_skill.utils.structs.pose import Pose
# from mani_skill.utils.structs.types import SimConfig
# from mani_skill.utils.structs import Actor

# # grab cube and return to rest keyframe
# @register_env("GrabCube-v1", max_episode_steps=50)
# class GrabCubeEnv(BaseEnv):
#     SUPPORTED_ROBOTS = ["koch-v1.1"]
#     agent: Koch
#     # spawn box params
#     spawn_box_half_size = 0.175
#     resting_tolerance = 0.1
#     # Domain Randomization params
#     # uniform dist to encourage robust policy
#     # TODO: (xhin) test normal dist in future
#     # small differing cube sizes
#     min_cube_half_size = 0.015
#     max_cube_half_size = 0.02

#     # min_cube_friction = 0.3
#     # max_cube_friction = 0.3

#     min_cube_friction = 1
#     max_cube_friction = 1

#     def __init__(
#         self, *args, robot_uids="koch-v1.1", robot_init_qpos_noise=0.02, reconfiguration_freq=1, num_envs=1,**kwargs
#     ):
#         self.robot_init_qpos_noise = robot_init_qpos_noise
#         reconfiguration_freq = 1
#         super().__init__(
#             *args,
#             robot_uids=robot_uids,
#             reconfiguration_freq=reconfiguration_freq,
#             num_envs=num_envs,
#             **kwargs,
#         )

#     @property
#     def _default_sensor_configs(self):
#         pose = sapien.Pose(
#             [-0.00347404, 0.136826, 0.496307],
#             [0.138651, 0.345061, 0.0516183, -0.926846],
#         )
#         return [CameraConfig("base_camera", pose, 640, 480, np.pi / 2, 0.01, 100)]

#     @property
#     def _default_human_render_camera_configs(self):
#         pose = sapien_utils.look_at(eye=[0.3, 0.4, 0.3], target=[-0.2, 0, 0.00])
#         return CameraConfig(
#             "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
#         )

#     def build_square_boundary(
#         self, size: float = 0.05, name: str = "square_boundary", color=[1, 1, 1, 1], initial_pose=sapien.Pose(p=[0,0,0]),
#     ):
#         builder = self.scene.create_actor_builder()
#         border_thickness = 0.01  # Adjust this value as needed
#         tape_thickness = 0.005
#         # Top border
#         builder.add_box_visual(
#             pose=sapien.Pose([0, size - border_thickness / 2, 0]),
#             half_size=[size, border_thickness / 2, tape_thickness / 2],
#             material=sapien.render.RenderMaterial(base_color=color),
#         )

#         # Bottom border
#         builder.add_box_visual(
#             pose=sapien.Pose([0, -size + border_thickness / 2, 0]),
#             half_size=[size, border_thickness / 2, tape_thickness / 2],
#             material=sapien.render.RenderMaterial(base_color=color),
#         )

#         # Left border
#         builder.add_box_visual(
#             pose=sapien.Pose([-size + border_thickness / 2, 0, 0]),
#             half_size=[border_thickness / 2, size, tape_thickness / 2],
#             material=sapien.render.RenderMaterial(base_color=color),
#         )

#         # Right border
#         builder.add_box_visual(
#             pose=sapien.Pose([size - border_thickness / 2, 0, 0]),
#             half_size=[border_thickness / 2, size, tape_thickness / 2],
#             material=sapien.render.RenderMaterial(base_color=color),
#         )
#         # setting initial pose
#         builder.initial_pose = initial_pose
#         return builder.build_kinematic(name)

#     # setting initial pose
#     def _load_agent(self, options: dict):
#         super()._load_agent(options, sapien.Pose(p=[0, 0, 0.2]))

#     def _load_scene(self, options: dict):
#         self.table_scene = TableSceneBuilder(
#             self, robot_init_qpos_noise=self.robot_init_qpos_noise
#         )

#         self.spawn_box  = self.build_square_boundary(
#             size=self.spawn_box_half_size, name="spawn_boundary", initial_pose=sapien.Pose(p=[-1,-1,2])
#         )

#         self.rest_qpos = torch.from_numpy(self.agent.keyframes["rest"].qpos).to(
#             self.device
#         )

#         self.table_scene.build()
#         cubes = []
#         half_sizes = []
#         # CUBE DR - size and friction
#         # TODO (xhin) - cube color DR
#         for i in range(self.num_envs):
#             builder = self.scene.create_actor_builder()
#             half_size = torch.rand(1).item() * (self.max_cube_half_size - self.min_cube_half_size) + self.min_cube_half_size
#             half_sizes.append(half_size)
#             friction = torch.rand(1).item() * (self.max_cube_friction - self.min_cube_friction) + self.min_cube_friction
#             material = sapien.pysapien.physx.PhysxMaterial(
#                 static_friction=friction,
#                 dynamic_friction=friction,
#                 restitution=0,
#             )
#             builder.add_box_collision(half_size=[half_size] * 3, material=material)
#             builder.add_box_visual(
#                 half_size=[half_size] * 3,
#                 material=sapien.render.RenderMaterial(
#                     base_color=[1,0,0,1],
#                 ),
#             )
#             # setting new pose for cube
#             builder.initial_pose = sapien.Pose(p=[0, 1, 2*half_size])
#             builder.set_scene_idxs([i])
#             cube = builder.build(f"cube_{i}")
#             cubes.append(cube)
#         self.cube = Actor.merge(cubes, "cube")
#         self.cube_half_sizes = torch.tensor(half_sizes).to(self.device)

#     def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
#         with torch.device(self.device):
#             b = len(env_idx)
#             self.table_scene.initialize(env_idx)

#             xyz = torch.zeros((b, 3))
#             # move box toward the arm
#             xyz[:, 0] = -0.175
#             self.spawn_box.set_pose(Pose.create_from_pq(p=xyz))

#             xyz = torch.zeros((b, 3))
#             xyz[:, :2] = (
#                 torch.rand((b, 2)) * self.spawn_box_half_size * 2
#                 - self.spawn_box_half_size
#             )
#             xyz[:, :2] += self.spawn_box.pose.p[env_idx, :2]
#             xyz[:, 2] = self.cube_half_sizes[env_idx]
#             qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
#             self.cube.set_pose(Pose.create_from_pq(xyz, qs))

#     def _get_obs_extra(self, info: Dict):
#         # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
#         obs = dict()
#         if "state" in self.obs_mode:
#             obs.update(
#                 obj_pose=self.cube.pose.raw_pose,
#                 tcp_to_obj_pos=self.cube.pose.p - (self.agent.tcp.pose.p+self.agent.tcp2.pose.p)/2,
#                 is_grasped=info["is_grasped"],
#                 tcp_pose=self.agent.tcp.pose.raw_pose,
#                 tcp2_pose = self.agent.tcp2.pose.raw_pose,
#                 cube_side_length=self.cube_half_sizes*2,
#                 grippers_distance=torch.linalg.norm(self.agent.tcp.pose.p - self.agent.tcp2.pose.p, axis=-1),
#             )
#         return obs

#     def evaluate(self):
#         is_grasped = self.agent.is_grasping(self.cube)
#         # don't include the grasping in requirement of end qpos
#         robot_to_rest_pose_dist = torch.linalg.norm(self.agent.robot.qpos[..., :-1] - self.rest_qpos[:-1], axis=1)
#         return {
#             "is_grasped": is_grasped,
#             "robot_to_grasped_rest_dist": robot_to_rest_pose_dist
#         }

#     def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
#         tcp_to_obj_dist = torch.linalg.norm(
#             self.cube.pose.p - (self.agent.tcp.pose.p+self.agent.tcp2.pose.p)/2, axis=1
#         )
#         # stage 1, reach tcp to object
#         reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
#         reward = reaching_reward

#         # stage 2, grasp the object
#         is_grasped = info["is_grasped"]
#         reward += is_grasped

#         # stage 2, return to rest position while grasping
#         reward += (1 - torch.tanh(info["robot_to_grasped_rest_dist"])) * is_grasped
#         return reward

#     def compute_normalized_dense_reward(
#         self, obs: Any, action: torch.Tensor, info: Dict
#     ):
#         return self.compute_dense_reward(obs=obs, action=action, info=info) / 3
