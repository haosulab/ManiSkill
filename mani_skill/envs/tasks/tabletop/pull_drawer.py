from typing import Dict, Union, Any
import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.types import SimConfig, GPUMemoryConfig
from mani_skill.sensors.camera import CameraConfig

@register_env("PullDrawer-v1", max_episode_steps=200)
class PullDrawerEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ("sparse", "dense", "normalized_dense", "none")
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        
        # Outer cabinet dimensions (scaled down by 0.5)
        self.outer_width = 0.15    # x (was 0.3)
        self.outer_depth = 0.2     # y (was 0.4)
        self.outer_height = 0.15   # z (was 0.3)
        self.wall_thickness = 0.01  # (was 0.02)
        
        # Inner drawer dimensions (slightly smaller to fit inside)
        self.inner_width = self.outer_width - 2.2 * self.wall_thickness
        self.inner_depth = self.outer_depth - 1.2 * self.wall_thickness
        self.inner_height = self.outer_height - 2.2 * self.wall_thickness
        
        # Handle dimensions (scaled down)
        self.handle_width = 0.04    # Width of handle bar
        self.handle_height = 0.02   # Height of handle from drawer face
        self.handle_thickness = 0.01  # Thickness of handle material
        self.handle_offset = 0.02   # Offset from drawer front (new parameter)
        
        # Movement parameters (scaled proportionally)
        self.max_pull_distance = self.outer_depth * 0.8  # Can pull out 80% of depth
        self.target_pos = -self.max_pull_distance * 0.8
        
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
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.5], target=[-0.1, 0, 0.1])
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
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return [
            CameraConfig(
                "render_camera",
                pose=pose,
                width=512,
                height=512,
                fov=1,
                near=0.01,
                far=100,
            )
        ]

    def _load_scene(self, options: dict):
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()

        builder = self.scene.create_articulation_builder()
        
        # Create outer cabinet (fixed base)
        base = builder.create_link_builder()
        base.name = "cabinet"
        
        # Bottom base
        base.add_box_collision(
            sapien.Pose([0, 0, -self.outer_height/2]),
            half_size=[self.outer_width/2, self.outer_depth/2, self.wall_thickness/2]
        )
        base.add_box_visual(
            sapien.Pose([0, 0, -self.outer_height/2]),
            half_size=[self.outer_width/2, self.outer_depth/2, self.wall_thickness/2],
        )
        
        # Top wall
        base.add_box_collision(
            sapien.Pose([0, 0, self.outer_height/2]),
            half_size=[self.outer_width/2, self.outer_depth/2, self.wall_thickness/2]
        )
        base.add_box_visual(
            sapien.Pose([0, 0, self.outer_height/2]),
            half_size=[self.outer_width/2, self.outer_depth/2, self.wall_thickness/2],
        )
        
        # Back wall
        base.add_box_collision(
            sapien.Pose([0, -self.outer_depth/2, 0]),
            half_size=[self.outer_width/2, self.wall_thickness/2, self.outer_height/2]
        )
        base.add_box_visual(
            sapien.Pose([0, -self.outer_depth/2, 0]),
            half_size=[self.outer_width/2, self.wall_thickness/2, self.outer_height/2],
        )
        
        # Side walls
        for x_sign in [-1, 1]:
            base.add_box_collision(
                sapien.Pose([x_sign * self.outer_width/2, 0, 0]),
                half_size=[self.wall_thickness/2, self.outer_depth/2, self.outer_height/2]
            )
            base.add_box_visual(
                sapien.Pose([x_sign * self.outer_width/2, 0, 0]),
                half_size=[self.wall_thickness/2, self.outer_depth/2, self.outer_height/2],
            )
        
        # Create sliding drawer
        drawer = builder.create_link_builder(parent=base)
        drawer.name = "drawer"
        
        # Drawer bottom
        drawer.add_box_collision(
            sapien.Pose([0, 0, -self.inner_height/2]),
            half_size=[self.inner_width/2, self.inner_depth/2, self.wall_thickness/2]
        )
        drawer.add_box_visual(
            sapien.Pose([0, 0, -self.inner_height/2]),
            half_size=[self.inner_width/2, self.inner_depth/2, self.wall_thickness/2],
        )
        
        # Drawer back
        drawer.add_box_collision(
            sapien.Pose([0, -self.inner_depth/2, 0]),
            half_size=[self.inner_width/2, self.wall_thickness/2, self.inner_height/2]
        )
        drawer.add_box_visual(
            sapien.Pose([0, -self.inner_depth/2, 0]),
            half_size=[self.inner_width/2, self.wall_thickness/2, self.inner_height/2],
        )
        
        # Drawer sides
        for x_sign in [-1, 1]:
            drawer.add_box_collision(
                sapien.Pose([x_sign * self.inner_width/2, 0, 0]),
                half_size=[self.wall_thickness/2, self.inner_depth/2, self.inner_height/2]
            )
            drawer.add_box_visual(
                sapien.Pose([x_sign * self.inner_width/2, 0, 0]),
                half_size=[self.wall_thickness/2, self.inner_depth/2, self.inner_height/2],
            )
        
        # Drawer front
        drawer.add_box_collision(
            sapien.Pose([0, self.inner_depth/2, 0]),
            half_size=[self.inner_width/2, self.wall_thickness/2, self.inner_height/2]
        )
        drawer.add_box_visual(
            sapien.Pose([0, self.inner_depth/2, 0]),
            half_size=[self.inner_width/2, self.wall_thickness/2, self.inner_height/2],
        )
        
        # Handle (horizontal bar) - Now positioned clearly in front of drawer
        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 0, 0, 1])
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0
        
        # Main handle bar
        drawer.add_box_collision(
            sapien.Pose([0, self.inner_depth/2 + self.handle_offset, 0]),
            half_size=[self.handle_width/2, self.handle_thickness/2, self.handle_thickness/2]
        )
        drawer.add_box_visual(
            sapien.Pose([0, self.inner_depth/2 + self.handle_offset, 0]),
            half_size=[self.handle_width/2, self.handle_thickness/2, self.handle_thickness/2],
            material=mat
        )
        
        # Handle supports (vertical bars connecting to drawer front)
        for x_sign in [-1, 1]:
            support_x = x_sign * (self.handle_width/2 - self.handle_thickness/2)
            drawer.add_box_collision(
                sapien.Pose([support_x, self.inner_depth/2 + self.handle_offset/2, 0]),
                half_size=[self.handle_thickness/2, self.handle_offset/2, self.handle_thickness/2]
            )
            drawer.add_box_visual(
                sapien.Pose([support_x, self.inner_depth/2 + self.handle_offset/2, 0]),
                half_size=[self.handle_thickness/2, self.handle_offset/2, self.handle_thickness/2],
                material=mat
            )

        # Configure drawer joint
        drawer.set_joint_properties(
            type="prismatic",
            limits=(-self.max_pull_distance, 0),
            pose_in_parent=sapien.Pose([0, 0, 0]),
            pose_in_child=sapien.Pose(),
            friction=0.1,
            damping=10
        )

        # Build and position the drawer
        builder.set_scene_idxs(scene_idxs=range(self.num_envs))
        builder.set_initial_pose(sapien.Pose(p=[-0.2, -0.5, 0.3]))  
        
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
        drawer_pulled = pos_dist.squeeze(-1) < 0.01
        
        progress = 1 - torch.tanh(5.0 * pos_dist)
        
        return {
            "success": drawer_pulled,
            "success_once": drawer_pulled,
            "success_at_end": drawer_pulled,
            "drawer_progress": progress.mean(),
            "drawer_distance": pos_dist.mean(),
            "reward": self.compute_normalized_dense_reward(
                None, None, {"success": drawer_pulled}
            ),
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        drawer_qpos = self.drawer.get_qpos()
        pos_dist = torch.abs(self.target_pos - drawer_qpos)
        
        # Get gripper position relative to handle
        tcp_pose = self.agent.tcp.pose.raw_pose
        tcp_pos = tcp_pose[..., :3]
        # Updated handle position calculation to match new handle position
        handle_pos = torch.tensor(
            [-0.2, -0.5 + self.inner_depth/2 + self.handle_offset, 0.3], 
            device=self.device
        )
        reach_dist = torch.norm(tcp_pos - handle_pos, dim=-1)
        
        # Compute rewards
        distance_reward = 2.0 * (1 - torch.tanh(5.0 * pos_dist)).squeeze(-1)
        reaching_reward = 1.0 * (1 - torch.tanh(10.0 * reach_dist))
        is_grasping = self.agent.is_grasping(self.drawer_link, max_angle=30)
        grasping_reward = 2.0 * is_grasping
        
        success_mask = info.get("success", torch.zeros_like(is_grasping))
        success_reward = torch.where(success_mask, 5.0, 0.0)
        
        return distance_reward + reaching_reward + grasping_reward + success_reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 10.0  # Maximum possible reward (2 + 1 + 2 + 5)
        dense_reward = self.compute_dense_reward(obs=obs, action=action, info=info)
        return dense_reward / max_reward