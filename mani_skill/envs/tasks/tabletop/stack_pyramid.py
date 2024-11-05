from typing import Any, Dict, Union

import numpy as np
import torch

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose  
from mani_skill.utils.logging_utils import logger

@register_env("StackPyramid-v1", max_episode_steps=50)
class StackPyramidEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    SUPPORTED_REWARD_MODES = ["sparse", "none"]
    
    agent: Union[Panda, Fetch]

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.num_cubes = 6 # 3 + 2 + 1
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
            self.scene, half_size=0.02, color=[0, 0, 1, 1], name="cubeC"
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
            cubeB_xy = xy + sampler.sample(radius, 100)
            cubeC_xy = xy + sampler.sample(radius, 100)

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
                lock_z=False
            )
            self.cubeC.set_pose(Pose.create_from_pq(p=xyz, q=qs))
        # ...

    def evaluate(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        pos_C = self.cubeC.pose.p

        offset_AB = pos_A - pos_B
        offset_BC = pos_B - pos_C
        offset_AC = pos_A - pos_C

        def evaluate_cube_distance(offset, cube_a, cube_b, top_or_next):
            tolerance = 0.5
            if top_or_next == "top":
                xy_offset = torch.linalg.norm(offset[..., :2], axis=-1) - torch.linalg.norm(self.cube_half_size[:2])
                z_offset = torch.linalg.norm(offset[..., 2]) - torch.linalg.norm(2 * self.cube_half_size[2])
                
                xy_flag = xy_offset <= tolerance
                z_flag = z_offset <= tolerance

                logger.debug(f"TOP Distance XY: {xy_offset}")
                logger.debug(f"TOP Distance Z: {z_offset}")
            else:
                xy_offset = torch.linalg.norm(offset[..., :2], axis=-1) - torch.linalg.norm(2 * self.cube_half_size[:2])
                z_offset = torch.abs(offset[..., 2] - self.cube_half_size[2])
                xy_flag = xy_offset <= 0.05
                z_flag = z_offset <= 0.05
                logger.debug(f"NEXT TO Distance XY: {xy_offset}")
                logger.debug("NEXT TO Distance Z:", z_offset)
            
            if (xy_flag == False):
                logger.debug("XY FLAG FALSE")
            if (z_flag == False):
                logger.debug("Z FLAG FALSE")
            is_cubeA_on_cubeB = torch.logical_and(xy_flag, z_flag)
            is_cubeA_static = cube_a.is_static(lin_thresh=1e-2, ang_thresh=0.5)
            is_cubeA_grasped = self.agent.is_grasping(cube_a)
            success = is_cubeA_on_cubeB & is_cubeA_static & (~is_cubeA_grasped)

            logger.debug("offset:", offset)
            logger.debug(f"Evaluating {cube_a} on {cube_b} ({top_or_next}):")
            logger.debug("  xy_flag:", xy_flag)
            logger.debug("  z_flag:", z_flag)
            logger.debug("  is_cubeA_on_cubeB:", is_cubeA_on_cubeB)
            logger.debug("  is_cubeA_static:", is_cubeA_static)
            logger.debug("  is_cubeA_grasped:", is_cubeA_grasped)
            logger.debug("  success:", success)

            return success.bool()

        success_A_B = evaluate_cube_distance(offset_AB, self.cubeA, self.cubeB, "next_to")
        success_C_B = evaluate_cube_distance(offset_BC, self.cubeC, self.cubeB, "top")
        success_C_A = evaluate_cube_distance(offset_AC, self.cubeC, self.cubeA, "top")
        
        success = success_A_B and success_C_B and success_C_A

        return {
            "success": success,
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
                cubeB_to_cubeC_pos=self.cubeC.pose.p - self.cubeB.pose.p,
                cubeA_to_cubeC_pos=self.cubeC.pose.p - self.cubeA.pose.p,
            )
        return obs
