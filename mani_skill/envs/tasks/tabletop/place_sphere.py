from typing import Any, Dict, Union

import numpy as np
import torch
import torch.random
import sapien
from transforms3d.euler import euler2quat
from mani_skill.envs.utils import randomization

from mani_skill.agents.robots import Fetch, Panda, Xmate3Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig

import matplotlib.pyplot as plt
import gymnasium as gym

@register_env("PlaceSphere-v1", max_episode_steps=50)
class PlaceSphereEnv(BaseEnv):
    """
    Task Description
    ----------------
    Place the sphere into the shallow bin.

    Randomizations
    --------------
    The position of the bin and the sphere are randomized: The bin is inited in [0, 0.1]x[-0.1, 0.1], and the sphere is inited in [-0.1, -0.05]x[-0.1, 0.1]
    
    Success Conditions
    ------------------
    The sphere is place on the top of the bin. The robot remains static and the gripper is not closed at the end state
    """

    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]

    # Specify some supported robot types
    agent: Union[Panda, Xmate3Robotiq, Fetch]

    # set some commonly used values
    radius = 0.02 # radius of the sphere
    inner_side_half_len = 0.02 # side length of the bin's inner square
    short_side_half_size = 0.0025 # length of the shortest edge of the block
    block_half_size = [short_side_half_size, 2*short_side_half_size+inner_side_half_len, 2*short_side_half_size+inner_side_half_len] # The bottom block of the bin, which is larger: The list represents the half length of the block along the [x, y, z] axis respectively.
    edge_block_half_size = [short_side_half_size, 2*short_side_half_size+inner_side_half_len, 2*short_side_half_size] # The edge block of the bin, which is smaller. The representations are similar to the above one
        
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_cfg=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.2], target=[-0.1, 0, 0])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, -0.2, 0.2], [0.0, 0.0, 0.2])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )
        
    def _build_bin(self, radius):
        builder = self.scene.create_actor_builder()
        
        # init the locations of the basic blocks
        dx = self.block_half_size[1] - self.block_half_size[0] 
        dy = self.block_half_size[1] - self.block_half_size[0] 
        dz = self.edge_block_half_size[2] + self.block_half_size[0]
        
        # build the bin bottom and edge blocks
        poses = [
            sapien.Pose([0, 0, 0]),
            sapien.Pose([-dx, 0, dz]),
            sapien.Pose([dx, 0, dz]),
            sapien.Pose([0, -dy, dz]),
            sapien.Pose([0, dy, dz]),
        ]
        half_sizes = [
            [self.block_half_size[1], self.block_half_size[2], self.block_half_size[0]],
            self.edge_block_half_size,
            self.edge_block_half_size,
            [self.edge_block_half_size[1], self.edge_block_half_size[0], self.edge_block_half_size[2]],
            [self.edge_block_half_size[1], self.edge_block_half_size[0], self.edge_block_half_size[2]],
        ]
        for pose, half_size in zip(poses, half_sizes):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size)

      	# build the kinematic bin
        return builder.build_kinematic(name="bin")

    def _load_scene(self, options: dict):
        # load the table
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        # load the sphere
        self.obj = actors.build_sphere(
            self.scene,
            radius=self.radius,
            color=np.array([12, 42, 160, 255]) / 255,
            name="sphere",
            body_type="dynamic",
        )
        
        # load the bin
        self.bin = self._build_bin(self.radius)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            # init the table scene
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            # init the sphere in the first 1/4 zone along the x-axis (so that it doesn't collide the bin)
            xyz = torch.zeros((b, 3))
            xyz[..., 0] = (torch.rand((b, 1)) * 0.05 - 0.1)[..., 0] # first 1/4 zone of x ([-0.1, -0.05])
            xyz[..., 1] = (torch.rand((b, 1)) * 0.2 - 0.1)[..., 0] # spanning all possible ys
            xyz[..., 2] = self.radius # on the table
            q = [1, 0, 0, 0]
            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.obj.set_pose(obj_pose)

            # init the bin in the last 1/2 zone along the x-axis (so that it doesn't collide the sphere)
            pos = torch.zeros((b, 3))
            pos[:, 0] = torch.rand((b, 1))[..., 0] * 0.1 # the last 1/2 zone of x ([0, 0.1])
            pos[:, 1] = torch.rand((b, 1))[..., 0] * 0.2 - 0.1 # spanning all possible ys
            pos[:, 2] = self.block_half_size[0] # on the table
            q = [1, 0, 0, 0]
            bin_pose = Pose.create_from_pq(p=pos, q=q)
            self.bin.set_pose(bin_pose)

    def evaluate(self):
        pos_obj = self.obj.pose.p
        pos_bin = self.bin.pose.p
        offset = pos_obj - pos_bin
        xy_flag = (
            torch.linalg.norm(offset[..., :2], axis=1)
            <= 0.005
        )
        z_flag = torch.abs(offset[..., 2] - self.radius - self.block_half_size[0]) <= 0.005
        is_obj_on_bin = torch.logical_and(xy_flag, z_flag)
        is_obj_static = self.obj.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_obj_grasped = self.agent.is_grasping(self.obj)
        success = is_obj_on_bin & is_obj_static & (~is_obj_grasped)
        return {
            "is_obj_grasped": is_obj_grasped,
            "is_obj_on_bin": is_obj_on_bin,
            "is_obj_static": is_obj_static,
            "success": success,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            is_grasped=info["is_obj_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            bin_pos=self.bin.pose.p
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.obj.pose.raw_pose,
                tcp_to_obj_pos=self.obj.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward
        tcp_pose = self.agent.tcp.pose.p
        obj_pos = self.obj.pose.p
        obj_to_tcp_dist = torch.linalg.norm(tcp_pose - obj_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * obj_to_tcp_dist))

        # grasp and place reward
        obj_pos = self.obj.pose.p
        bin_pos = self.bin.pose.p
        bin_top_pos = self.bin.pose.p.clone()
        bin_top_pos[:, 2] = bin_top_pos[:, 2] + self.block_half_size[0] + self.radius
        obj_to_bin_top_dist = torch.linalg.norm(bin_top_pos - obj_pos, axis=1)
        place_reward = 1 - torch.tanh(5.0 * obj_to_bin_top_dist)
        reward[info["is_obj_grasped"]] = (4 + place_reward)[info["is_obj_grasped"]]

        # ungrasp and static reward
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(
            self.device
        )
        is_obj_grasped = info["is_obj_grasped"]
        ungrasp_reward = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward[~is_obj_grasped] = 16.0 # give ungrasp a bigger reward, so that it exceeds the robot static reward and the gripper can close
        v = torch.linalg.norm(self.obj.linear_velocity, axis=1)
        av = torch.linalg.norm(self.obj.angular_velocity, axis=1)
        static_reward = 1 - torch.tanh(v * 10 + av)
        robot_static_reward = self.agent.is_static(0.2) # keep the robot static at the end state, since the sphere may spin when being placed on top
        reward[info["is_obj_on_bin"]] = (
            6 + (ungrasp_reward + static_reward + robot_static_reward) / 3.0
        )[info["is_obj_on_bin"]]
        
        # success reward
        reward[info["success"]] = 13
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 13.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward




