from typing import Any, Dict

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.agents.robots.koch.koch import Koch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.digital_twins.base_env import BaseDigitalTwinEnv
from mani_skill.envs.tasks.digital_twins.utils.camera_randomization import (
    make_camera_rectangular_prism,
    noised_look_at,
)
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig, SimConfig


# grab cube and return to rest keyframe
@register_env("GrabCube-v1", max_episode_steps=75)
class GrabCubeEnv(BaseDigitalTwinEnv):
    # TODO(xhin): make a better interface for choosing camera positions, e.g. using yaml file
    # cameras
    base_camera_pos = [-0.15, -0.16, 0.44]
    max_camera_offset = [1e-4, 1e-4, 1e-4]
    camera_target = [-0.38 - 0.03, -0.829, 0.08]
    camera_target_noise = 0.0
    camera_view_rot_noise = 0.0

    # robot
    SUPPORTED_ROBOTS = ["koch-v1.1"]
    agent: Koch
    rgb_overlay_paths = dict(base_camera="obs_img_new.png")
    dist_from_table_edge = 0.5

    # Task DR
    spawn_box_half_size = 0.1

    cube_size_mean = 0.0175
    cube_size_std = 7e-4

    cube_friction_mean = 0.3
    cube_friction_std = 0.05

    def __init__(
        self,
        *args,
        robot_uids="koch-v1.1",
        robot_init_qpos_noise=0.02,
        num_envs=1,
        cam_rand_on_step=True,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.cam_rand_on_step = cam_rand_on_step
        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            enable_shadow=False,
            **kwargs,
        )

    def get_random_camera_pose(self):
        eyes = make_camera_rectangular_prism(
            self.num_envs,
            scale=self.max_camera_offset,
            center=self.base_camera_pos,
            theta=0,
        )
        return noised_look_at(
            eyes,
            target=self.camera_target,
            look_at_noise=self.camera_target_noise,
            view_axis_rot_noise=self.camera_view_rot_noise,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**22, max_rigid_patch_count=2**21
            ),
            sim_freq=120,
            control_freq=30,
        )

    @property
    def _default_sensor_configs(self):
        # mount does all reposing, allows the user to easily toggle randomizing cameras per step
        return CameraConfig(
            "base_camera",
            pose=Pose.create_from_pq(p=[0, 0, 0]),
            width=128,
            height=128,
            fov=1,
            near=0.01,
            far=100,
            mount=self.camera_mount,
        )

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig(
            "render_camera",
            pose=self.init_cam_poses,
            width=512,
            height=512,
            fov=1,
            near=0.01,
            far=100,
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0.2]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )

        self.rest_qpos = (
            torch.from_numpy(self.agent.keyframes["rest"].qpos).to(self.device).float()
        )

        self.table_scene.build()
        cubes = []
        half_sizes = []
        # CUBE DR: size, friction, and color
        # ROBOT DR: color
        sampled_sizes = torch.normal(
            mean=torch.ones(self.num_envs) * self.cube_size_mean,
            std=torch.ones(self.num_envs) * self.cube_size_std,
        )
        sampled_frictions = torch.normal(
            mean=torch.ones(self.num_envs) * self.cube_friction_mean,
            std=torch.ones(self.num_envs) * self.cube_friction_std,
        )
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            half_size = sampled_sizes[i].item()
            friction = sampled_frictions[i].clip(0.1, 0.5).item()
            half_sizes.append(half_size)
            material = sapien.pysapien.physx.PhysxMaterial(
                static_friction=friction,
                dynamic_friction=friction,
                restitution=0,
            )
            builder.add_box_collision(
                half_size=[half_size] * 3, material=material, density=25
            )
            color = list(torch.rand(3))
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

        cam_builder = self.scene.create_actor_builder()
        cam_builder.initial_pose = sapien.Pose(p=[0.1, 0.1, 0.1])
        self.camera_mount = cam_builder.build_kinematic("camera_mount")

        self.init_cam_poses = self.get_random_camera_pose()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            # move coordinate system to [-table_edge_x,0,0]
            table_edge_x = (
                self.table_scene.table.pose.p[0, 0].abs().item()
                + self.table_scene.table_length / 2
            )
            table_edge_y = (
                self.table_scene.table.pose.p[0, 0].abs().item()
                + self.table_scene.table_width / 2
            )

            # first set the robot to offset from edge of table
            robot_pose = self.agent.robot.pose
            robot_pose.p[..., 1] = (
                -(table_edge_y - self.dist_from_table_edge)
                + self.table_scene.table.pose.p[..., 1]
            )
            self.agent.robot.set_pose(robot_pose)

            # set spawnbox
            spawn_box_pos = robot_pose.p + torch.tensor([0.45, 0, 0])

            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                torch.rand((b, 2)) * self.spawn_box_half_size * 2
                - self.spawn_box_half_size
            )
            xyz[:, :2] += spawn_box_pos[env_idx, :2]
            xyz[:, 2] = self.cube_half_sizes[env_idx]
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            # set camera mount init episode pose
            # if cam_rand_on_step is false, this will be the only place camera randomizaiton occurs
            self.camera_mount.set_pose(self.get_random_camera_pose())

    def _before_control_step(self):
        if self.cam_rand_on_step:
            self.camera_mount.set_pose(self.get_random_camera_pose())
            if self.gpu_sim_enabled:
                self.scene.px.gpu_apply_rigid_dynamic_data()
                self.scene.px.gpu_fetch_rigid_dynamic_data()

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict()
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p
                - (self.agent.tcp.pose.p + self.agent.tcp2.pose.p) / 2,
                is_grasped=info["is_grasped"],
                is_properly_grapsed=info["is_properly_grasped"],
                grippers_distance=info["grippers_distance"],
                tcp_pose=self.agent.tcp.pose.raw_pose,
                tcp2_pose=self.agent.tcp2.pose.raw_pose,
                cube_side_length=self.cube_half_sizes * 2,
            )
        obs.update(
            to_rest_dist=self.rest_qpos[:-1] - self.agent.robot.qpos[..., :-1],
            rest_qpos=self.rest_qpos[:-1].view(1, 5).repeat(self.num_envs, 1),
            target_qpos=self.agent.controller._target_qpos.clone(),
        )
        return obs

    def evaluate(self):
        touching_table = self.agent._compute_undesired_contacts(self.table_scene.table)
        is_grasped = self.agent.is_grasping(self.cube)
        grippers_distance = torch.linalg.norm(
            self.agent.tcp.pose.p - self.agent.tcp2.pose.p, axis=-1
        )
        is_properly_grasped = is_grasped * (
            grippers_distance >= (2 * self.cube_half_sizes)
        )
        robot_to_rest_pose_dist = torch.linalg.norm(
            (self.agent.controller._target_qpos.clone()[..., :-1] / (2 * np.pi))
            - (self.rest_qpos[:-1] / (2 * np.pi)),
            axis=1,
        )
        success = (robot_to_rest_pose_dist < 0.05) * is_grasped
        return {
            "is_grasped": is_grasped,
            "grippers_distance": grippers_distance,
            "robot_to_grasped_rest_dist": robot_to_rest_pose_dist,
            "touching_table": touching_table,
            "is_properly_grasped": is_properly_grasped,
            "success": success,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        is_grasped = info["is_grasped"]
        is_properly_grasped = info["is_properly_grasped"]
        gripper_finger_dist = torch.linalg.norm(
            self.agent.tcp.pose.p - self.agent.tcp2.pose.p, axis=-1
        )
        tcp_pos = (self.agent.tcp.pose.p + self.agent.tcp2.pose.p) / 2

        # stage 1, reach tcp to object
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - tcp_pos,
            axis=-1,
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        # still stage 1, orient gripper correctly, important for correctly grasping the
        # we want the between finger vectors to be perpendicular to the up vector of the cube, reward when dot product is zero
        finger2_to_finger1_unitvec = (
            self.agent.tcp.pose.p - self.agent.tcp2.pose.p
        ) / gripper_finger_dist.unsqueeze(-1)
        orientation_reward = 1 - (finger2_to_finger1_unitvec[..., -1]).abs().view(
            self.num_envs
        )
        reward += (
            orientation_reward * ~is_grasped  # * ~is_properly_grasped
        )  # only reward for orienting when cube is not grasped

        # stage 2, reward for properly grasping the cube, + additional 1 to replace the lost orientation reward
        reward += 2 * is_properly_grasped.float()

        # stage 3, lift the cube
        cube_lifted = (
            self.cube.pose.p[..., -1] >= (self.cube_half_sizes + 1e-3)
        ) * is_grasped  # * is_properly_grasped

        # reward += cube_lifted.float()
        # cube_zdist = ((self.cube.pose.p[..., -1] - self.cube_half_sizes) - 0.04).abs()
        cube_zdist = ((self.cube.pose.p[..., -1] - self.cube_half_sizes) - 0.05).abs()
        reward += (
            1 - torch.tanh(20 * cube_zdist)
        ) * is_grasped  # * is_properly_grasped

        # stage 4 alternative
        reward += (
            8 * (1 - torch.tanh(4 * info["robot_to_grasped_rest_dist"])) * cube_lifted
        )

        # negative reward: closing gripper for no reason
        closed_gripper = gripper_finger_dist <= ((4 / 5) * 2 * self.cube_half_sizes)
        closed_gripper_rew = closed_gripper * (
            1 - (gripper_finger_dist / (2 * self.cube_half_sizes))
        )

        return (reward - 2 * info["touching_table"]) - closed_gripper_rew

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 12
