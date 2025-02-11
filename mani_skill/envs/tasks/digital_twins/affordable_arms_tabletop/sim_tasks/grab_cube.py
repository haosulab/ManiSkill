from typing import Any, Dict

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots.koch.koch import Koch
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig, SimConfig

from .sim4real_base_env import Sim4RealBaseEnv


# grab cube and return to rest keyframe
@register_env("GrabCube-v1", max_episode_steps=75)
class GrabCubeEnv(Sim4RealBaseEnv):
    # Task DR
    spawn_box_half_size = 0.1 / 2

    cube_size_mean = 0.0175 / 2
    cube_size_std = 7e-4 / 2

    cube_friction_mean = 0.3
    cube_friction_std = 0.05

    rand_cube_color = True

    noise_qpos = True

    def __init__(
        self,
        *args,
        robot_uids="koch-v1.1",
        robot_init_qpos_noise=0.02,
        num_envs=1,
        cam_rand_on_step=True,
        keyframe_id=None,
        enable_shadow=False,
        **kwargs,
    ):
        super().__init__(
            *args,
            robot_uids=robot_uids,
            robot_init_qpos_noise=robot_init_qpos_noise,
            num_envs=num_envs,
            cam_rand_on_step=cam_rand_on_step,
            keyframe_id=keyframe_id,
            enable_shadow=enable_shadow,
            **kwargs,
        )

    def _load_scene(self, options: dict):
        super()._load_scene(options)

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
                half_size=[half_size] * 3, material=material, density=200  # 25
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

        self.rest_qpos = (
            torch.from_numpy(self.agent.keyframes["elevated_turn"].qpos)
            .to(self.device)
            .float()
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            # self.agent.robot.set_pose(Pose.create_from_pq(p=self.agent.robot.pose.p, q=euler2quat(0, 0, np.pi / 2)))

            xyz = torch.zeros((b, 3))
            spawn_box_pos = self.agent.robot.pose.p + torch.tensor([0.225, 0, 0])

            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                torch.rand((b, 2)) * self.spawn_box_half_size * 2
                - self.spawn_box_half_size
            )
            xyz[:, :2] += spawn_box_pos[env_idx, :2]
            xyz[:, 2] = self.cube_half_sizes[env_idx]
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

    def _get_obs_extra(self, info: Dict):
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

        # automatically included state info
        qpos = self.agent.robot.get_qpos().clone()
        qpos_noise = (
            0  # torch.normal(torch.zeros_like(qpos), torch.ones_like(qpos)*0.015)
        )
        qpos += qpos_noise
        obs.update(
            qpos=qpos,
            to_rest_dist=self.rest_qpos[:-1] - qpos[..., :-1],
            rest_qpos=self.rest_qpos[:-1].view(1, 5).repeat(self.num_envs, 1),
            target_qpos=self.agent.controller._target_qpos.clone(),
            is_grasped=info[
                "is_grasped"
            ],  # large target vs actual qpos in real robot determine is_grasped in real
        )
        return obs

    def evaluate(self):
        touching_table = self.agent._compute_undesired_contacts(self.table_scene.table)
        # calculate if the actual agent would grasp the cube here - from empirical testing of robot gripper and cubesize
        is_grasped = self.agent.is_grasping(self.cube)
        # end calculate if the actual agent would grasp the cube
        grippers_distance = torch.linalg.norm(
            self.agent.tcp.pose.p - self.agent.tcp2.pose.p, axis=-1
        )
        tcp_pos = (self.agent.tcp.pose.p + self.agent.tcp2.pose.p) / 2
        # TODO look into smoother alternative for easier prediction of this, currently binary
        tcp_isclose = (
            torch.linalg.norm(tcp_pos - self.cube.pose.p, dim=-1)
            <= self.cube_half_sizes
        )
        # is_properly_grasped = is_grasped * tcp_isclose

        is_properly_grasped = is_grasped * (
            grippers_distance >= (2 * self.cube_half_sizes * 0.99)
        )  #################################################### changed

        robot_to_rest_pose_dist = torch.linalg.norm(
            (self.agent.controller._target_qpos.clone()[..., :-1] / (np.pi))
            - (self.rest_qpos[:-1] / (np.pi)),
            axis=1,
        )

        cube_lifted = (
            self.cube.pose.p[..., -1] >= (self.cube_half_sizes + 1e-3)
        ) * is_grasped  # * is_properly_grasped
        success = cube_lifted
        return {
            "is_grasped": is_grasped,
            "grippers_distance": grippers_distance,
            "robot_to_grasped_rest_dist": robot_to_rest_pose_dist,
            "touching_table": touching_table,
            "tcp_isclose": tcp_isclose,
            "is_properly_grasped": is_properly_grasped,
            "cube_lifted": cube_lifted,
            "success": success,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # state properties
        is_properly_grasped = info["is_properly_grasped"]
        is_close = info["tcp_isclose"]
        gripper_finger_dist = torch.linalg.norm(
            self.agent.tcp.pose.p - self.agent.tcp2.pose.p, axis=-1
        )
        tcp_pos = (self.agent.tcp.pose.p + self.agent.tcp2.pose.p) / 2

        finger2_to_finger1_unitvec = (
            self.agent.tcp.pose.p - self.agent.tcp2.pose.p
        ) / gripper_finger_dist.unsqueeze(-1)
        cube_lifted = info["cube_lifted"]

        # reward
        reward = 0

        # stage 1, reach tcp to object
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - tcp_pos,
            axis=-1,
        )
        # reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reaching_reward = 1 - torch.tanh(15 * tcp_to_obj_dist)
        reward += reaching_reward

        # still stage 1, orient gripper correctly, important for correctly grasping the
        # we want the between finger vectors to be perpendicular to the z vector, reward when dot product is zero
        orientation_reward = 1 - (finger2_to_finger1_unitvec[..., -1]).abs().view(
            self.num_envs
        )
        reward += orientation_reward

        # stage 2, grasp the cube
        # close_gripper_dist = self.agent.controller._target_qpos.clone()[..., -1].abs() # closed at 0
        close_gripper_dist = (2 * (self.cube_half_sizes) - gripper_finger_dist).abs()
        reward += 2 * (1 - torch.tanh(40 * close_gripper_dist)) * is_close.float()
        reward += is_properly_grasped.float()

        # stage 3, lift the cube
        reward += cube_lifted.float()

        # stage 4 return to rest position
        reward += (
            3 * (1 - torch.tanh(4 * info["robot_to_grasped_rest_dist"])) * cube_lifted
        )

        # just don't close gripper early, it's that simple :)
        gripper_closing = self.agent.robot.qpos[..., -1] <= 0.9
        reward -= 3 * gripper_closing * torch.tanh(10 * tcp_to_obj_dist) * ~is_close

        # touch table
        reward -= 2.0 * info["touching_table"].float()

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 7
