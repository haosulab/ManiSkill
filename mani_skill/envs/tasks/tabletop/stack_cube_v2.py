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
@register_env("StackCube-v2", max_episode_steps=100)
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


    def evaluate(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        pos_C = self.cubeC.pose.p

        offset_AB = pos_A - pos_B
        offset_CA = pos_C - pos_A

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


        is_cubeA_static = self.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_cubeC_static = self.cubeC.is_static(lin_thresh=1e-2, ang_thresh=0.5)

        is_cubeA_grasped = self.agent.is_grasping(self.cubeA)
        is_cubeC_grasped = self.agent.is_grasping(self.cubeC)
        task_1_done = is_cubeA_on_cubeB * is_cubeA_static * (~is_cubeA_grasped)
        task_2_done = is_cubeC_on_cubeA * is_cubeC_static * (~is_cubeC_grasped)
        success = (
            task_1_done * task_2_done
        )
        
        return {
            "is_cubeA_grasped": is_cubeA_grasped,
            "is_cubeA_on_cubeB": is_cubeA_on_cubeB,
            "is_cubeA_static": is_cubeA_static,
            "is_cubeC_grasped": is_cubeC_grasped,
            "is_cubeC_on_cubeA": is_cubeC_on_cubeA,
            "is_cubeC_static": is_cubeC_static,
            "task_1_done": task_1_done.bool(),
            "task_2_done": task_2_done.bool(),
            "success": success.bool(),
        }


    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=self.cubeB.pose.raw_pose,
                cubeC_pose=self.cubeC.pose.raw_pose,
                tcp_to_cubeA_pos=self.cubeA.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeB_pos=self.cubeB.pose.p - self.agent.tcp.pose.p,
                tcp_to_cubeC_pos=self.cubeC.pose.p - self.agent.tcp.pose.p,
                cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
                cubeC_to_cubeA_pos=self.cubeA.pose.p - self.cubeC.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        
        def _compute_reward(cube_1, cube_2, is_cube1_grasped, is_cube1_static,is_cube1_on_cube2,task_done):
            reward=torch.zeros_like(info["success"],dtype=torch.float32,device=self.device)
            # reaching reward
            tcp_pose = self.agent.tcp.pose.p
            cube_1_pos = cube_1.pose.p
            cube_1_to_tcp_dist = torch.linalg.norm(tcp_pose - cube_1_pos, axis=1)
            reward = 2 * (1 - torch.tanh(5 * cube_1_to_tcp_dist))

            # grasp and place reward
            cube_1_pos = cube_1.pose.p
            cube_2_pos = cube_2.pose.p
            goal_xyz = torch.hstack(
                [cube_2_pos[:, 0:2], (cube_2_pos[:, 2] + self.cube_half_size[2] * 2)[:, None]]
            )
            cube_1_to_goal_dist = torch.linalg.norm(goal_xyz - cube_1_pos, axis=1)
            place_reward = 1 - torch.tanh(5.0 * cube_1_to_goal_dist)

            reward[info[is_cube1_grasped]] = (4 + place_reward)[info[is_cube1_grasped]]

            # ungrasp and static reward
            gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(
                self.device
            )
            is_cube1_grasped = info[is_cube1_grasped]
            ungrasp_reward = (
                torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
            )
            ungrasp_reward[~is_cube1_grasped] = 1.0
            v = torch.linalg.norm(cube_1.linear_velocity, axis=1)
            av = torch.linalg.norm(cube_1.angular_velocity, axis=1)
            static_reward = 1 - torch.tanh(v * 10 + av)
            reward[info[is_cube1_on_cube2]] = (
                6 + (ungrasp_reward + static_reward) / 2.0
            )[info[is_cube1_on_cube2]]
            reward[info[task_done]] =8.0
            return reward
        task_1_reward=_compute_reward(self.cubeA,self.cubeB,"is_cubeA_grasped","is_cubeA_static","is_cubeA_on_cubeB","task_1_done")
        task_2_reward=_compute_reward(self.cubeC,self.cubeA,"is_cubeC_grasped","is_cubeC_static","is_cubeC_on_cubeA","task_2_done")
        return task_1_reward+9*task_2_reward*info["task_1_done"]

        

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 80.0