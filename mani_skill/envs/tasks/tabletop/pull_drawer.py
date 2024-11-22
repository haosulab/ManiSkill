from typing import Dict, Union, Any
import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import SimConfig, GPUMemoryConfig

@register_env("PullDrawer-v1", max_episode_steps=200)
class PullDrawerEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ("sparse", "dense", "normalized_dense", "none")
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    
    drawer_half_size = [0.1, 0.15, 0.01]
    drawer_body_half_size = [0.09, 0.14, 0.05]
    target_pos = -0.125
    max_pull_distance = 0.15

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(
            *args,
            robot_uids=robot_uids,
            **kwargs
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, 
                max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0.2, -0.3, 0.3], [0, 0, 0.1])
        return [
            CameraConfig(
                "base_camera", 
                pose=pose, 
                width=128, 
                height=128, 
                fov=np.pi / 2,
                near=0.01,
                far=100
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-0.4, -0.5, 0.6], [0.0, 0.0, 0.2])
        return [
            CameraConfig(
                "render_camera", 
                pose=pose, 
                width=512, 
                height=512, 
                fov=1,
                near=0.01,
                far=100
            )
        ]

    def _load_scene(self, options: dict):
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()

        # Create drawer
        builder = self.scene.create_articulation_builder()
        
        # Base/outer casing with walls
        base = builder.create_link_builder()
        base.name = "drawer_base"
        wall_thickness = 0.01

        # Back wall
        base.add_box_collision(
            sapien.Pose([-(self.drawer_half_size[0] - wall_thickness/2), 0, 0]),
            [wall_thickness/2, self.drawer_half_size[1], self.drawer_half_size[2]]
        )
        base.add_box_visual(
            sapien.Pose([-(self.drawer_half_size[0] - wall_thickness/2), 0, 0]),
            [wall_thickness/2, self.drawer_half_size[1], self.drawer_half_size[2]]
        )

        # Side walls
        for y_sign in [-1, 1]:
            base.add_box_collision(
                sapien.Pose([0, y_sign * (self.drawer_half_size[1] - wall_thickness/2), 0]),
                [self.drawer_half_size[0], wall_thickness/2, self.drawer_half_size[2]]
            )
            base.add_box_visual(
                sapien.Pose([0, y_sign * (self.drawer_half_size[1] - wall_thickness/2), 0]),
                [self.drawer_half_size[0], wall_thickness/2, self.drawer_half_size[2]]
            )

        # Bottom wall
        base.add_box_collision(
            sapien.Pose([0, 0, -(self.drawer_half_size[2] - wall_thickness/2)]),
            [self.drawer_half_size[0], self.drawer_half_size[1], wall_thickness/2]
        )
        base.add_box_visual(
            sapien.Pose([0, 0, -(self.drawer_half_size[2] - wall_thickness/2)]),
            [self.drawer_half_size[0], self.drawer_half_size[1], wall_thickness/2]
        )

        # Inner sliding drawer
        drawer = builder.create_link_builder(parent=base)
        drawer.name = "drawer_body"
        drawer.add_box_collision(half_size=self.drawer_body_half_size)
        drawer.add_box_visual(half_size=self.drawer_body_half_size)

        # Add handle
        handle_size = [0.02, 0.04, 0.01]
        handle_offset = 0.02
        drawer.add_box_collision(
            sapien.Pose([self.drawer_body_half_size[0] + handle_offset, 0, 0]),
            handle_size
        )
        drawer.add_box_visual(
            sapien.Pose([self.drawer_body_half_size[0] + handle_offset, 0, 0]),
            handle_size
        )

        drawer.set_joint_properties(
            type="prismatic",
            limits=(-self.max_pull_distance, 0),
            pose_in_parent=sapien.Pose([0, 0, 0]),
            pose_in_child=sapien.Pose(),
            friction=0.1,
            damping=10
        )

        # Build articulation with drawer slightly ajar
        builder.set_scene_idxs(scene_idxs=range(self.num_envs))
        builder.set_initial_pose(sapien.Pose(p=[0.12, 0, 0.2]))
        
        self.drawer = builder.build(fix_root_link=True, name="drawer_articulation")
        self.drawer_link = self.drawer.get_links()[1]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)
            
            init_pos = torch.zeros((b, 1), device=self.device)
            noise = torch.rand((b, 1), device=self.device) * 0.02
            init_pos -= noise
            
            self.drawer.set_qpos(init_pos)

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                drawer_pose=self.drawer.pose.raw_pose,
                drawer_qpos=self.drawer.get_qpos(),
            )
        return obs

    def evaluate(self):
        drawer_qpos = self.drawer.get_qpos()
        pos_dist = torch.abs(self.target_pos - drawer_qpos)
        cube_pulled_close = pos_dist.squeeze(-1) < 0.007
        
        progress = 1 - torch.tanh(5.0 * pos_dist)
        
        return {
            "success": cube_pulled_close,
            "success_once": cube_pulled_close,
            "success_at_end": cube_pulled_close,
            "drawer_progress": progress.mean(),
            "drawer_distance": pos_dist.mean(),
            "reward": self.compute_normalized_dense_reward(
                None, None, {"success": cube_pulled_close}
            ),
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        drawer_qpos = self.drawer.get_qpos()
        pos_dist = torch.abs(self.target_pos - drawer_qpos)
        
        distance_reward = 2.0 * (1 - torch.tanh(5.0 * pos_dist)).squeeze(-1)
        
        is_grasping = self.agent.is_grasping(self.drawer_link, max_angle=30)
        grasping_reward = 2.0 * is_grasping
        
        success_mask = info.get("success", torch.zeros_like(is_grasping))
        success_reward = torch.where(success_mask, 5.0, 0.0)
        
        return distance_reward + grasping_reward + success_reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 5.0
        dense_reward = self.compute_dense_reward(obs=obs, action=action, info=info)
        return dense_reward / max_reward
