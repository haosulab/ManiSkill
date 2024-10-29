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
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


# grab cube and return to rest keyframe
@register_env("GrabCube-v1", max_episode_steps=75)
class GrabCubeEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["koch-v1.1"]
    agent: Koch
    # spawn box params
    spawn_box_half_size = 0.175
    resting_tolerance = 0.1
    # Domain Randomization params
    # uniform dist to encourage robust policy
    # TODO: (xhin) test normal dist in future
    # small differing cube sizes
    min_cube_half_size = 0.015
    max_cube_half_size = 0.02

    # min_cube_friction = 0.3
    # max_cube_friction = 0.3

    min_cube_friction = 0.2
    max_cube_friction = 1

    mean_action_mag = 0

    cube_goal_pos = [-0.34, 0, 0.1]

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
        reconfiguration_freq = 1
        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            enable_shadow=True,
            **kwargs,
        )

    # @property
    # def _default_sim_config(self):
    #     return SimConfig(
    #         gpu_memory_config=GPUMemoryConfig(
    #             found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
    #         )
    #     )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=[0.3 - 0.071, 0.4 - 0.047, 0.3 + 0.05], target=[-0.2, 0, 0.1]
        )
        return CameraConfig(
            "base_camera", pose=pose, width=128, height=128, fov=1, near=0.01, far=100
        )

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0.4, 0.3], target=[-0.2, 0, 0.00])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def build_square_boundary(
        self,
        size: float = 0.05,
        name: str = "square_boundary",
        color=[1, 1, 1, 1],
        initial_pose=sapien.Pose(p=[0, 0, 0]),
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
        # setting initial pose
        builder.initial_pose = initial_pose
        return builder.build_kinematic(name)

    # setting initial pose
    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0.2]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )

        self.spawn_box = self.build_square_boundary(
            size=self.spawn_box_half_size,
            name="spawn_boundary",
            initial_pose=sapien.Pose(p=[-1, -1, 2]),
        )

        self.rest_qpos = (
            torch.from_numpy(self.agent.keyframes["rest"].qpos).to(self.device).float()
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
            color = np.random.rand(3)
            builder.add_box_visual(
                half_size=[half_size] * 3,
                material=sapien.render.RenderMaterial(
                    base_color=[color[0], color[1], color[2], 1],
                ),
            )
            # setting new pose for cube
            builder.initial_pose = sapien.Pose(p=[0, 1, 2 * half_size])
            builder.set_scene_idxs([i])
            cube = builder.build(f"cube_{i}")
            cubes.append(cube)
        self.cube = Actor.merge(cubes, "cube")
        self.cube_half_sizes = torch.tensor(half_sizes).to(self.device)

        # print("mean action mag", self.mean_action_mag)
        if isinstance(self.cube_goal_pos, list):
            self.cube_goal_pos = torch.tensor(
                self.cube_goal_pos, device=self.device
            ).float()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            # move box toward the arm
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
                # goal=self.cube_goal_pos.view(1, -1).repeat(self.num_envs, 1),
                goal=self.rest_qpos.view(1, -1).repeat(self.num_envs, 1).float(),
            )
        return obs

    def evaluate(self):
        is_grasped = self.agent.is_grasping(self.cube)
        # don't include the grasping in requirement of end qpos
        # robot_to_rest_pose_dist = torch.linalg.norm(self.agent.robot.qpos[..., :-1] - self.rest_qpos[:-1], axis=1)
        # robot_to_rest_pose_dist = ((self.agent.robot.qpos[..., :-1] / (2*np.pi)) - (self.rest_qpos[:-1] / (2*np.pi))).abs().mean(dim=-1)
        robot_to_rest_pose_dist = torch.linalg.norm(
            (self.agent.robot.qpos[..., :-1] / (2 * np.pi))
            - (self.rest_qpos[:-1] / (2 * np.pi)),
            axis=1,
        )

        touching_table = self.agent._compute_undesired_contacts(self.table_scene.table)
        return {
            "is_grasped": is_grasped,
            "robot_to_grasped_rest_dist": robot_to_rest_pose_dist,
            "touching_table": touching_table,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - (self.agent.tcp.pose.p + self.agent.tcp2.pose.p) / 2,
            axis=1,
        )
        # stage 1, reach tcp to object
        reaching_reward = 1 - torch.tanh(
            10 * tcp_to_obj_dist
        )  # reaching_reward = 1 - torch.tanh(10 * tcp_to_obj_dist)
        reward = reaching_reward

        # stage 2, grasp the object
        is_grasped = info["is_grasped"]
        reward += is_grasped.float()

        # stage 3, lift the cube
        cube_lifted = self.cube.pose.p[..., -1] >= (self.cube_half_sizes + 1e-3)
        reward += cube_lifted.float()

        # stage 4, demotivate flipping the cube on x or y axis before grapsing
        # can rotate around z axis, but other axes off limits
        # no_rot_reward = 1 - (
        #     torch.tanh(torch.linalg.norm(self.cube.angular_velocity[..., :-1], axis=1))
        #     * ~is_grasped
        # )
        pre_rot_rew = (
            torch.tanh(torch.linalg.norm(self.cube.angular_velocity[..., :-1], axis=1))
            * ~is_grasped
        )

        # stage 5, return to rest position while grasping and lifting cube
        reward += (
            9
            * (1 - torch.tanh(4 * info["robot_to_grasped_rest_dist"]))
            * is_grasped
            * cube_lifted
        )

        # return (reward * ~info["touching_table"])  * (no_rot_reward) - (torch.linalg.norm(action, axis=1) / (10))
        # works well
        # return ((reward - 0.1*info["touching_table"])  - 0.1*pre_rot_rew) - 0.1*torch.linalg.norm(action, axis=1)
        return (
            (reward - 0.1 * info["touching_table"]) - 0.1 * pre_rot_rew
        ) - 0.2 * torch.linalg.norm(action, axis=1)
        # return ((reward * ~info["touching_table"])  * (1-pre_rot_rew)) - 0.1*torch.linalg.norm(action, axis=1)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 12
