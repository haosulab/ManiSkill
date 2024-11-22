from typing import Dict, List, Union
import numpy as np
import sapien
import torch
from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, io_utils, sapien_utils
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig

@register_env("PullDrawer-v1", max_episode_steps=200)
class PullDrawerEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["sparse", "none"]
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        reconfiguration_freq=None,
        num_envs=1,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0
            
        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([-0.4, 0, 0.3], [0, 0, 0.1])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2)
        ]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()

        # Create a simple drawer using ArticulationBuilder
        builder = self.scene.create_articulation_builder()
        
        # Create the base (fixed part)
        base = builder.create_link_builder()
        base.add_box_collision(half_size=[0.2, 0.3, 0.02])
        base.add_box_visual(half_size=[0.2, 0.3, 0.02], color=[0.8, 0.8, 0.8, 1])
        
        # Create the drawer (moving part)
        drawer = builder.create_link_builder(parent=base)
        drawer.add_box_collision(half_size=[0.18, 0.28, 0.1])
        drawer.add_box_visual(half_size=[0.18, 0.28, 0.1], color=[0.6, 0.6, 0.6, 1])
        

        drawer.set_joint_properties(
            type="prismatic",
            limits=(-0.3, 0),  # 30cm travel range
            pose_in_parent=sapien.Pose([0, 0, 0.1]), 
            pose_in_child=sapien.Pose(),  
            friction=0.1,
            damping=10
        )
        drawer.set_joint_name("drawer_joint")

        # Build the articulation
        builder.set_initial_pose(sapien.Pose(p=[0, 0, 0.5]))  
        self.drawer = builder.build(fix_root_link=True)  
        
        self.drawer_link = self.drawer.get_links()[1]  
        self.target_pos = -0.25  # set target position as 25cm out

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.scene_builder.initialize(env_idx)
            # Reset drawer position
            self.drawer.set_qpos([0])  # Close the drawer initially

    @property
    def current_pos(self):
        return self.drawer.get_qpos()[0]

    def evaluate(self):
        pos_dist = self.target_pos - self.current_pos
        return dict(
            success=abs(pos_dist) < 0.02,  # 2cm threshold
            pos_dist=pos_dist
        )

    def _get_obs_extra(self, info: Dict):
        return dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            drawer_pos=self.current_pos,
            target_pos=self.target_pos,
            drawer_link_pos=self.drawer_link.pose.p
        )

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        reward = 0.0
        
        # Success bonus
        if info["success"]:
            return 10.0
        
        # Distance reward
        pos_dist = abs(info["pos_dist"])
        reward += 1 - np.tanh(pos_dist * 5.0)
        
        # Contact reward
        is_contacted = any(self.agent.check_contact_fingers(self.drawer_link))
        if is_contacted:
            reward += 0.25
        
        # Movement reward based on position difference
        pos_diff = info["pos_dist"] - self.last_pos_dist
        if info["pos_dist"] > 0:
            # Penalize moving away from target
            movement_reward = -np.tanh(pos_diff * 2) * 5
        else:
            # Reward moving toward target
            movement_reward = np.tanh(pos_diff * 2) * 5
        reward += movement_reward
        
        # Store current distance for next step
        self.last_pos_dist = info["pos_dist"]
        
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        max_reward = 10.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward