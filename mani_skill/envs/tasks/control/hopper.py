"""Adapted from https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/hopper.py"""
import os
from typing import Any, Dict, Union

import numpy as np
import torch

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization, rewards
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import (
    Array,
    GPUMemoryConfig,
    SceneConfig,
    SimConfig,
)

# fix new imports later
from mani_skill.utils.building.ground import build_ground
from transforms3d.euler import euler2quat

MJCF_FILE = f"{os.path.join(os.path.dirname(__file__), 'assets/hopper.xml')}"

# params from dm_control
# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 0.6

# Hopping speed above which hop reward is 1.
_HOP_SPEED = 2

class HopperRobot(BaseAgent):
    uid = "hopper"
    mjcf_path = MJCF_FILE

    @property
    def _sensor_configs(self):
        return [
            # replicated from xml file
            CameraConfig(
                uid="cam0",
                pose=Pose.create_from_pq([0, -2.8, 0], euler2quat(0,0,np.pi/2)),
                width=512,
                height=512,
                fov=70*(np.pi/180),
                near=0.01,
                far=100,
                entity_uid="torso_dummy_1",
            ),
            # CameraConfig(
            #     uid="back",
            #     pose=Pose.create_from_pq([-2, -0.2, -1.2], [1, 0, 0, 0]),
            #     width=512,
            #     height=512,
            #     fov=np.pi/2,
            #     near=0.01,
            #     far=100,
            #     entity_uid="torso_dummy_0",
            # ),
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        # NOTE joints in [rootx,rooty,rooz] are for planar tracking, not control joints
        pd_joint_delta_pos = PDJointPosControllerConfig(
            [j.name for j in self.robot.active_joints if 'root' not in j.name],
            -1,
            1,
            damping=5,
            stiffness=20,
            force_limit=100,
            use_delta=True,
        )
        rest = PassiveControllerConfig([j.name for j in self.robot.active_joints if 'root' in j.name], damping=0, friction=0)
        return deepcopy_dict(
            dict(
                pd_joint_delta_pos=dict(
                    body=pd_joint_delta_pos, rest=rest, balance_passive_force=False
                ),
            )
        )

    def _load_articulation(self):
        """
        Load the robot articulation
        """
        loader = self.scene.create_mjcf_loader()
        asset_path = str(self.mjcf_path)

        loader.name = self.uid

        # only need the robot
        self.robot = loader.parse(asset_path)[0][0].build()
        assert self.robot is not None, f"Fail to load URDF/MJCF from {asset_path}"

        # Cache robot link ids
        self.robot_link_ids = [link.name for link in self.robot.get_links()]

class HopperEnv(BaseEnv):
    agent: Union[HopperRobot]

    def __init__(self, *args, robot_uids=HopperRobot, **kwargs):
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
        return self.agent._sensor_configs

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0, -2, 0.5], target=[0, 0, 0.5])
        return [CameraConfig("render_camera", pose, 512, 512, (70/180)*np.pi, 0.01, 100)]
        # return self.agent._sensor_configs

    def _load_scene(self, options: dict):
        loader = self.scene.create_mjcf_loader()
        articulation_builders, actor_builders, sensor_configs = loader.parse(MJCF_FILE)
        for a in actor_builders:
            a.build(a.name)
        
        # load in the ground
        self.ground = build_ground(
            self.scene, floor_width=2, floor_length=50, altitude=-0.075, floor_origin=(25-2,0)
        )
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        with torch.device(self.device):
            b = len(env_idx)
            # qpos sampled same as dm_control, but ensure no self intersection explicitly here
            random_qpos = torch.rand(b, self.agent.robot.dof[0])
            q_lims = self.agent.robot.get_qlimits()
            q_ranges = q_lims[..., 1] - q_lims[..., 0]
            random_qpos *= q_ranges
            random_qpos += q_lims[..., 0]

            # overwrite planar joint qpos - these are special for planar robots
            # first two joints are dummy rootx and rootz
            random_qpos[:,:2] = 0
            # y is axis of rotation of our planar robot (xz plane), so we randomize around it
            random_qpos[:,2] = torch.pi*(2*torch.rand(b) - 1) # (-pi,pi)
            self.agent.robot.set_qpos(random_qpos)

    def evaluate(self):
        return dict()
    
    #dm_control function
    def touch(self, link_name):
        """Returns function of sensor force values"""
        force_vec = self.agent.robot.get_net_contact_forces([link_name])
        force_mag = torch.linalg.norm(force_vec, dim=-1)
        return torch.log1p(force_mag)
    
    #dm_control function
    @property
    def height(self):
        """Returns relative height of the robot"""
        return (self.agent.robot.links_map['torso'].pose.p[:,-1] - 
                self.agent.robot.links_map['foot_heel'].pose.p[:,-1]).view(-1,1)

    # overwrite basic obs state dict
    # same as dm_control does, need to remove inclusion of rootx joint qpos
    # obs is a 15 dim vec of qpos, qvel, and touch info
    def _get_obs_state_dict(self, info: Dict):
        return dict(
            qpos=self.agent.robot.get_qpos()[:,1:],
            qvel=self.agent.robot.get_qvel(),
            toe_touch=self.touch("foot_toe"),
            heel_touch=self.touch("foot_heel"),
        )

@register_env("MS-HopperStand-v1", max_episode_steps=100)
class HopperStandEnv(HopperEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        rew = rewards.tolerance(self.height, lower=_STAND_HEIGHT, upper=2.0)
        #print("rewards", rew.shape)
        return rew.view(-1)

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
    
# @register_env("MS-HopperHop-v1", max_episode_steps=1000)
# class HopperStandEnv(HopperEnv):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
    
#     def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
#         return 0

#     def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
#         # this should be equal to compute_dense_reward / max possible reward
#         max_reward = 1.0
#         return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward