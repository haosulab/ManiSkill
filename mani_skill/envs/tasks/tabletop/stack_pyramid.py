from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose  
from mani_skill.utils.logging_utils import logger

@register_env("StackPyramid-v1", max_episode_steps=250)
class StackPyramidEnv(BaseEnv):
    """
    **Task Description:**
    - The goal is to pick up a red cube, place it next to the green cube, and stack the blue cube on top of the red and green cube without it falling off.

    **Randomizations:**
    - all cubes have their z-axis rotation randomized
    - all cubes have their xy positions on top of the table scene randomized. The positions are sampled such that the cubes do not collide with each other

    **Success Conditions:**
    - the blue cube is static
    - the blue cube is on top of both the red and green cube (to within half of the cube size)
    - none of the red, green, blue cubes are grasped by the robot (robot must let go of the cubes)

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/StackPyramid-v1_rt.mp4"

    """

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    SUPPORTED_REWARD_MODES = ["none", "sparse"]
    
    agent: Union[Panda, Fetch]

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[-0.5,0.0,0.25], target=[0.2,0.0,-0.5])
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
            self.scene, half_size=0.02, color=[1, 0, 0, 1], name="cubeA", initial_pose=sapien.Pose(p=[0, 0, 0.2])
        )
        self.cubeB = actors.build_cube(
            self.scene, half_size=0.02, color=[0, 1, 0, 1], name="cubeB", initial_pose=sapien.Pose(p=[1, 0, 0.2])
        )
        self.cubeC = actors.build_cube(
            self.scene, half_size=0.02, color=[0, 0, 1, 1], name="cubeC", initial_pose=sapien.Pose(p=[-1, 0, 0.2])
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            region = [[-0.1, -0.2], [0.1, 0.2]]
            cubeA_xy, cubeB_xy, cubeC_xy = sample_safe_offsets(b=b, sample_region=region, min_distance=0.03, max_iter=100, device=self.device)

            # Cube A
            xyz = torch.zeros((b, 3), device=self.device)
            xyz[:, 2] = 0.02  # table height offset
            xyz[:, :2] = cubeA_xy
            
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )

            self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            # Cube B
            xyz[:, :2] = cubeB_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeB.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))
            
            # Cube C
            xyz[:, :2] = cubeC_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeC.set_pose(Pose.create_from_pq(p=xyz, q=qs))

    def evaluate(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        pos_C = self.cubeC.pose.p

        offset_AB = pos_A - pos_B
        offset_BC = pos_B - pos_C
        offset_AC = pos_A - pos_C

        def evaluate_cube_distance(offset, cube_a, cube_b, top_or_next):
            xy_flag = (torch.linalg.norm(offset[..., :2], axis=1) 
                       <= torch.linalg.norm(2*self.cube_half_size[:2]) 
                       + 0.005
                       )
            z_flag = torch.abs(offset[..., 2]) > 0.02
            if top_or_next == "top":
                is_cubeA_on_cubeB = torch.logical_and(xy_flag, z_flag)
            elif top_or_next == "next_to":
                is_cubeA_on_cubeB = xy_flag
            else:
                return NotImplementedError(f"Expect top_or_next to be either 'top' or 'next', got {top_or_next}")
            
            is_cubeA_static = cube_a.is_static(lin_thresh=1e-2, ang_thresh=0.5)
            is_cubeA_grasped = self.agent.is_grasping(cube_a)

            success = is_cubeA_on_cubeB & is_cubeA_static & (~is_cubeA_grasped)            
            return success.bool()

        success_A_B = evaluate_cube_distance(offset_AB, self.cubeA, self.cubeB, "next_to")
        success_C_B = evaluate_cube_distance(offset_BC, self.cubeC, self.cubeB, "top")
        success_C_A = evaluate_cube_distance(offset_AC, self.cubeC, self.cubeA, "top")
        success = torch.logical_and(success_A_B, torch.logical_and(success_C_B, success_C_A))
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

def sample_safe_offsets(b: int, sample_region: Union[float, int, list], min_distance: float = 0.05, max_iter: int = 100, device: torch.device = torch.device("cpu")):
    """
    For each sample in the batch, sample three (2,) offsets such that
    all pairwise distances are at least min_distance.

    sample_region can either be:
      - a float/int that specifies the sample region's side length. This assumes the sample region is a square, or 
      - a list defining the coordinate region as [[low_x, low_y], [high_x, high_y]]
      
    Returns three tensors of shape (b, 2) for offsets A, B, and C.
    """
    if isinstance(sample_region, (list)):
        low = torch.tensor(sample_region[0], device=device)
        high = torch.tensor(sample_region[1], device=device)
        distribution = torch.distributions.Uniform(low, high)
    elif isinstance(sample_region, (float, int)):
        half = sample_region / 2.0
        low = torch.tensor([-half, -half], device=device)
        high = torch.tensor([half, half], device=device)
        distribution = torch.distributions.Uniform(low, high)
    else:
        raise ValueError("sample_region must be either a float/int or a list of two coordinate pairs, e.g. [[low_x, low_y], [high_x, high_y]]")
    
    offsets_A = torch.zeros((b, 2), device=device)
    offsets_B = torch.zeros((b, 2), device=device)
    offsets_C = torch.zeros((b, 2), device=device)
    for i in range(b):
        for iter_count in range(max_iter):
            off = distribution.sample((3,))
            d_AB = torch.norm(off[0] - off[1])
            d_AC = torch.norm(off[0] - off[2])
            d_BC = torch.norm(off[1] - off[2])
            offsets_A[i] = off[0]
            offsets_B[i] = off[1]
            offsets_C[i] = off[2]
            
            if d_AB >= min_distance and d_AC >= min_distance and d_BC >= min_distance:
                break
        else:
            print("Warning: no valid sample after max_iter iterations for sample index", i)
        
    return offsets_A, offsets_B, offsets_C