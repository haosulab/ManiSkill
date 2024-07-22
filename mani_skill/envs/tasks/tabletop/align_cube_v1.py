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
@register_env("AlignCube-v1", max_episode_steps=50)
class AlignCubeEnv_v1(BaseEnv):

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
        offset_BC = pos_B - pos_C
        offset_CD = pos_C - pos_D

        # Check if cubes are in a row (aligned horizontally with minimal vertical displacement)
        xy_flag_AB = torch.abs(offset_AB[..., 1]) <= 0.005
        z_flag_AB = torch.abs(offset_AB[..., 2]) <= 0.005
        is_cubeA_next_to_cubeB = torch.logical_and(xy_flag_AB, z_flag_AB)

        xy_flag_BC = torch.abs(offset_BC[..., 1]) <= 0.005
        z_flag_BC = torch.abs(offset_BC[..., 2]) <= 0.005
        is_cubeB_next_to_cubeC = torch.logical_and(xy_flag_BC, z_flag_BC)

        xy_flag_CD = torch.abs(offset_CD[..., 1]) <= 0.005
        z_flag_CD = torch.abs(offset_CD[..., 2]) <= 0.005
        is_cubeC_next_to_cubeD = torch.logical_and(xy_flag_CD, z_flag_CD)

        is_cubeA_static = self.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_cubeB_static = self.cubeB.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_cubeC_static = self.cubeC.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_cubeD_static = self.cubeD.is_static(lin_thresh=1e-2, ang_thresh=0.5)

        is_cubeA_grasped = self.agent.is_grasping(self.cubeA)
        is_cubeB_grasped = self.agent.is_grasping(self.cubeB)
        is_cubeC_grasped = self.agent.is_grasping(self.cubeC)
        is_cubeD_grasped = self.agent.is_grasping(self.cubeD)

        success = (
            is_cubeA_next_to_cubeB
            * is_cubeB_next_to_cubeC
            * is_cubeC_next_to_cubeD
            * is_cubeA_static
            * is_cubeB_static
            * is_cubeC_static
            * is_cubeD_static
            * (~is_cubeA_grasped)
            * (~is_cubeB_grasped)
            * (~is_cubeC_grasped)
            * (~is_cubeD_grasped)
        )

        return {
            "is_cubeA_grasped": is_cubeA_grasped,
            "is_cubeA_next_to_cubeB": is_cubeA_next_to_cubeB,
            "is_cubeA_static": is_cubeA_static,
            "is_cubeB_grasped": is_cubeB_grasped,
            "is_cubeB_next_to_cubeC": is_cubeB_next_to_cubeC,
            "is_cubeB_static": is_cubeB_static,
            "is_cubeC_grasped": is_cubeC_grasped,
            "is_cubeC_next_to_cubeD": is_cubeC_next_to_cubeD,
            "is_cubeC_static": is_cubeC_static,
            "is_cubeD_grasped": is_cubeD_grasped,
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
    # Position and alignment reward
        reward = torch.zeros_like(info["success"], dtype=torch.float32)
        
        # Reward for each cube being close to its target position in a row
        for cube, next_cube, key in zip([self.cubeA, self.cubeB, self.cubeC], [self.cubeB, self.cubeC, self.cubeD], ["A", "B", "C"]):
            pos_cube = cube.pose.p
            pos_next_cube = next_cube.pose.p
            dist = torch.linalg.norm(pos_cube[..., :2] - pos_next_cube[..., :2], axis=1)
            reward += 2 - torch.tanh(5 * dist)

        # Additional reward for alignment
        reward[info["is_cubeA_next_to_cubeB"]] += 2
        reward[info["is_cubeB_next_to_cubeC"]] += 2
        reward[info["is_cubeC_next_to_cubeD"]] += 2

        # Reward for all cubes being static and not grasped
        reward[info["is_cubeA_static"] & info["is_cubeB_static"] & info["is_cubeC_static"] & info["is_cubeD_static"]] += 1
        reward[~info["is_cubeA_grasped"] & ~info["is_cubeB_grasped"] & ~info["is_cubeC_grasped"] & ~info["is_cubeD_grasped"]] += 1

        # Final success reward
        reward[info["success"]] = 14

        return reward


    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 14.0
