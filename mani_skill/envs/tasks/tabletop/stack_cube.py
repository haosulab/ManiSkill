from typing import Any, Dict, Union

import numpy as np
import torch

from mani_skill.agents.robots import Fetch, Panda, Xmate3Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("StackCube-v1", max_episode_steps=50)
class StackCubeEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]
    agent: Union[Panda, Xmate3Robotiq, Fetch]

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cubeA = actors.build_cube(
            self.scene, half_size=0.02, color=[1, 0, 0, 1], name="cubeA"
        )
        self.cubeB = actors.build_cube(
            self.scene, half_size=0.02, color=[0, 1, 0, 1], name="cubeB"
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02
            xy = torch.rand((b, 2)) * 0.2 - 0.1
            region = [[-0.1, -0.2], [0.1, 0.2]]
            sampler = randomization.UniformPlacementSampler(bounds=region, batch_size=b)
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            cubeA_xy = xy + sampler.sample(radius, 100)
            cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)

            xyz[:, :2] = cubeA_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = cubeB_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeB.set_pose(Pose.create_from_pq(p=xyz, q=qs))

    def evaluate(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        offset = pos_A - pos_B
        xy_flag = (
            torch.linalg.norm(offset[..., :2], axis=1)
            <= torch.linalg.norm(self.cube_half_size[:2]) + 0.005
        )
        z_flag = torch.abs(offset[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.005
        is_cubeA_on_cubeB = torch.logical_and(xy_flag, z_flag)
        # NOTE (stao): GPU sim can be fast but unstable. Angular velocity is rather high despite it not really rotating
        is_cubeA_static = self.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_cubeA_grasped = self.agent.is_grasping(self.cubeA)
        success = is_cubeA_on_cubeB * is_cubeA_static * (~is_cubeA_grasped)
        return {
            "is_cubeA_grasped": is_cubeA_grasped,
            "is_cubeA_on_cubeB": is_cubeA_on_cubeB,
            "is_cubeA_static": is_cubeA_static,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=self.cubeB.pose.raw_pose,
                tcp_to_cubeA_pos=self.cubeA.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeB_pos=self.cubeB.pose.p - self.agent.tcp.pose.p,
                cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward
        tcp_pose = self.agent.tcp.pose.p
        cubeA_pos = self.cubeA.pose.p
        cubeA_to_tcp_dist = torch.linalg.norm(tcp_pose - cubeA_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * cubeA_to_tcp_dist))

        # grasp and place reward
        cubeA_pos = self.cubeA.pose.p
        cubeB_pos = self.cubeB.pose.p
        goal_xyz = torch.hstack(
            [cubeB_pos[:, 0:2], (cubeB_pos[:, 2] + self.cube_half_size[2] * 2)[:, None]]
        )
        cubeA_to_goal_dist = torch.linalg.norm(goal_xyz - cubeA_pos, axis=1)
        place_reward = 1 - torch.tanh(5.0 * cubeA_to_goal_dist)

        reward[info["is_cubeA_grasped"]] = (4 + place_reward)[info["is_cubeA_grasped"]]

        # ungrasp and static reward
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(
            self.device
        )  # NOTE: hard-coded with panda
        is_cubeA_grasped = info["is_cubeA_grasped"]
        ungrasp_reward = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward[~is_cubeA_grasped] = 1.0
        v = torch.linalg.norm(self.cubeA.linear_velocity, axis=1)
        av = torch.linalg.norm(self.cubeA.angular_velocity, axis=1)
        static_reward = 1 - torch.tanh(v * 10 + av)
        reward[info["is_cubeA_on_cubeB"]] = (
            6 + (ungrasp_reward + static_reward) / 2.0
        )[info["is_cubeA_on_cubeB"]]

        reward[info["success"]] = 8

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8


@register_env("StackCube-v2", max_episode_steps=50)
class StackCubeEnv_v2(BaseEnv):

    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]
    agent: Union[Panda, Xmate3Robotiq, Fetch]

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cubeA = actors.build_cube(
            self.scene, half_size=0.02, color=[1, 0, 0, 1], name="cubeA"
        )
        self.cubeB = actors.build_cube(
            self.scene, half_size=0.02, color=[0, 1, 0, 1], name="cubeB"
        )
        self.cubeC = actors.build_cube(
            self.scene, half_size=0.02, color=[1, 0, 1, 1], name="cubeC"
        )
        self.cubeD = actors.build_cube(
            self.scene, half_size=0.02, color=[0, 0, 1, 1], name="cubeD"
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02
            xy = torch.rand((b, 2)) * 0.2 - 0.1
            region = [[-0.1, -0.2], [0.1, 0.2]]
            sampler = randomization.UniformPlacementSampler(bounds=region, batch_size=b)
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            
            cubeA_xy = xy + sampler.sample(radius, 100)
            cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)
            cubeC_xy = xy + sampler.sample(radius, 100, verbose=False)
            cubeD_xy = xy + sampler.sample(radius, 100, verbose=False)

            xyz[:, :2] = cubeA_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = cubeB_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeB.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = cubeC_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeC.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = cubeD_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeD.set_pose(Pose.create_from_pq(p=xyz, q=qs))


    def evaluate(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        pos_C = self.cubeC.pose.p
        pos_D = self.cubeD.pose.p

        offset_AB = pos_A - pos_B
        offset_CA = pos_C - pos_A
        offset_DC = pos_D - pos_C

        xy_flag_AB = (
            torch.linalg.norm(offset_AB[..., :2], axis=1)
            <= torch.linalg.norm(self.cube_half_size[:2]) + 0.005
        )
        z_flag_AB = torch.abs(offset_AB[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.005
        is_cubeA_on_cubeB = torch.logical_and(xy_flag_AB, z_flag_AB)

        xy_flag_CA = (
            torch.linalg.norm(offset_CA[..., :2], axis=1)
            <= torch.linalg.norm(self.cube_half_size[:2]) + 0.005
        )
        z_flag_CA = torch.abs(offset_CA[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.005
        is_cubeC_on_cubeA = torch.logical_and(xy_flag_CA, z_flag_CA)

        xy_flag_DC = (
            torch.linalg.norm(offset_DC[..., :2], axis=1)
            <= torch.linalg.norm(self.cube_half_size[:2]) + 0.005
        )
        z_flag_DC = torch.abs(offset_DC[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.005
        is_cubeD_on_cubeC = torch.logical_and(xy_flag_DC, z_flag_DC)

        is_cubeA_static = self.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_cubeC_static = self.cubeC.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_cubeD_static = self.cubeD.is_static(lin_thresh=1e-2, ang_thresh=0.5)

        is_cubeA_grasped = self.agent.is_grasping(self.cubeA)
        is_cubeC_grasped = self.agent.is_grasping(self.cubeC)
        is_cubeD_grasped = self.agent.is_grasping(self.cubeD)

        success = (
            is_cubeA_on_cubeB 
            * is_cubeC_on_cubeA 
            * is_cubeD_on_cubeC 
            * is_cubeA_static 
            * is_cubeC_static 
            * is_cubeD_static 
            * (~is_cubeA_grasped) 
            * (~is_cubeC_grasped) 
            * (~is_cubeD_grasped)
        )
        
        return {
            "is_cubeA_grasped": is_cubeA_grasped,
            "is_cubeA_on_cubeB": is_cubeA_on_cubeB,
            "is_cubeA_static": is_cubeA_static,
            "is_cubeC_grasped": is_cubeC_grasped,
            "is_cubeC_on_cubeA": is_cubeC_on_cubeA,
            "is_cubeC_static": is_cubeC_static,
            "is_cubeD_grasped": is_cubeD_grasped,
            "is_cubeD_on_cubeC": is_cubeD_on_cubeC,
            "is_cubeD_static": is_cubeD_static,
            "success": success.bool(),
        }


    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=self.cubeB.pose.raw_pose,
                cubeC_pose=self.cubeC.pose.raw_pose,
                cubeD_pose=self.cubeD.pose.raw_pose,
                tcp_to_cubeA_pos=self.cubeA.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeB_pos=self.cubeB.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeC_pos=self.cubeC.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeD_pos=self.cubeD.pose.p - self.agent.tcp.pose.p,
                cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
                cubeC_to_cubeA_pos=self.cubeA.pose.p - self.cubeC.pose.p,
                cubeD_to_cubeC_pos=self.cubeC.pose.p - self.cubeD.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward
        tcp_pose = self.agent.tcp.pose.p
        cubeA_pos = self.cubeA.pose.p
        cubeA_to_tcp_dist = torch.linalg.norm(tcp_pose - cubeA_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * cubeA_to_tcp_dist))

        # grasp and place reward for cubeA
        cubeB_pos = self.cubeB.pose.p
        goal_xyz_A = torch.hstack(
            [cubeB_pos[:, 0:2], (cubeB_pos[:, 2] + self.cube_half_size[2] * 2)[:, None]]
        )
        cubeA_to_goal_dist = torch.linalg.norm(goal_xyz_A - cubeA_pos, axis=1)
        place_reward_A = 1 - torch.tanh(5.0 * cubeA_to_goal_dist)

        reward[info["is_cubeA_grasped"]] = (4 + place_reward_A)[info["is_cubeA_grasped"]]

        # ungrasp and static reward for cubeA
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(self.device)  # NOTE: hard-coded with panda
        is_cubeA_grasped = info["is_cubeA_grasped"]
        ungrasp_reward_A = torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        ungrasp_reward_A[~is_cubeA_grasped] = 1.0
        v_A = torch.linalg.norm(self.cubeA.linear_velocity, axis=1)
        av_A = torch.linalg.norm(self.cubeA.angular_velocity, axis=1)
        static_reward_A = 1 - torch.tanh(v_A * 10 + av_A)
        reward[info["is_cubeA_on_cubeB"]] = (6 + (ungrasp_reward_A + static_reward_A) / 2.0)[info["is_cubeA_on_cubeB"]]

        # grasp and place reward for cubeC
        cubeC_pos = self.cubeC.pose.p
        goal_xyz_C = torch.hstack(
            [cubeA_pos[:, 0:2], (cubeA_pos[:, 2] + self.cube_half_size[2] * 2)[:, None]]
        )
        cubeC_to_goal_dist = torch.linalg.norm(goal_xyz_C - cubeC_pos, axis=1)
        place_reward_C = 1 - torch.tanh(5.0 * cubeC_to_goal_dist)

        reward[info["is_cubeC_grasped"]] = (4 + place_reward_C)[info["is_cubeC_grasped"]]

        # ungrasp and static reward for cubeC
        is_cubeC_grasped = info["is_cubeC_grasped"]
        ungrasp_reward_C = torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        ungrasp_reward_C[~is_cubeC_grasped] = 1.0
        v_C = torch.linalg.norm(self.cubeC.linear_velocity, axis=1)
        av_C = torch.linalg.norm(self.cubeC.angular_velocity, axis=1)
        static_reward_C = 1 - torch.tanh(v_C * 10 + av_C)
        reward[info["is_cubeC_on_cubeA"]] = (6 + (ungrasp_reward_C + static_reward_C) / 2.0)[info["is_cubeC_on_cubeA"]]

        # grasp and place reward for cubeD
        cubeD_pos = self.cubeD.pose.p
        goal_xyz_D = torch.hstack(
            [cubeC_pos[:, 0:2], (cubeC_pos[:, 2] + self.cube_half_size[2] * 2)[:, None]]
        )
        cubeD_to_goal_dist = torch.linalg.norm(goal_xyz_D - cubeD_pos, axis=1)
        place_reward_D = 1 - torch.tanh(5.0 * cubeD_to_goal_dist)

        reward[info["is_cubeD_grasped"]] = (4 + place_reward_D)[info["is_cubeD_grasped"]]

        # ungrasp and static reward for cubeD
        is_cubeD_grasped = info["is_cubeD_grasped"]
        ungrasp_reward_D = torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        ungrasp_reward_D[~is_cubeD_grasped] = 1.0
        v_D = torch.linalg.norm(self.cubeD.linear_velocity, axis=1)
        av_D = torch.linalg.norm(self.cubeD.angular_velocity, axis=1)
        static_reward_D = 1 - torch.tanh(v_D * 10 + av_D)
        reward[info["is_cubeD_on_cubeC"]] = (6 + (ungrasp_reward_D + static_reward_D) / 2.0)[info["is_cubeD_on_cubeC"]]

        reward[info["success"]] = 8

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8