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
from mani_skill.utils.structs import Pose
from mani_skill.utils.building import actors

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
        
        # Outer cabinet dimensions 
        self.outer_width = 0.225    
        self.outer_depth = 0.3     
        self.outer_height = 0.225   
        self.wall_thickness = 0.03  
        
        # Inner drawer dimensions 
        self.inner_width = self.outer_width - 2 * self.wall_thickness
        self.inner_depth = self.outer_depth - 2.1 * self.wall_thickness
        self.inner_height = self.outer_height - 2.1 * self.wall_thickness
        
        # Handle dimensions 
        self.handle_width = 0.18    # Width of handle bar
        self.handle_height = 0.06   # Height of handle from drawer face
        self.handle_thickness = 0.015  # Thickness of handle material
        self.handle_offset = 0.11   # Offset from drawer side
        
        # Movement parameters 
        self.max_pull_distance = self.outer_width * 0.8  # Can pull out 80% of width
        self.target_pos = -self.max_pull_distance * 0.8
        self.k = 0.005
        
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
        pose = sapien_utils.look_at([-0.8, 0.7, 0.7], [0.0, 0.0, 0.0])
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
        
        # Create outer cabinet 
        base = builder.create_link_builder()
        base.set_name('cabinet')
        
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
        
        # Left wall 
        base.add_box_collision(
            sapien.Pose([0, -self.outer_depth/2, 0]),
            half_size=[self.outer_width/2, self.wall_thickness/2, self.outer_height/2]
        )
        base.add_box_visual(
            sapien.Pose([0, -self.outer_depth/2, 0]),
            half_size=[self.outer_width/2, self.wall_thickness/2, self.outer_height/2],
        )
        
        # Left wall 
        base.add_box_collision(
            sapien.Pose([0, self.outer_depth/2, 0]),
            half_size=[self.outer_width/2, self.wall_thickness/2, self.outer_height/2]
        )
        base.add_box_visual(
            sapien.Pose([0, self.outer_depth/2, 0]),
            half_size=[self.outer_width/2, self.wall_thickness/2, self.outer_height/2],
        )
        
        # Right wall 
        base.add_box_collision(
            sapien.Pose([self.outer_width/2, 0, 0]),
            half_size=[self.wall_thickness/2, self.outer_depth/2, self.outer_height/2]
        )
        base.add_box_visual(
            sapien.Pose([self.outer_width/2, 0, 0]),
            half_size=[self.wall_thickness/2, self.outer_depth/2, self.outer_height/2],
        )
        
        # Create sliding drawer
        drawer = builder.create_link_builder(parent=base)
        drawer.set_name('drawer')
        
        # Drawer bottom
        drawer.add_box_collision(
            sapien.Pose([0, 0, -self.inner_height/2]),
            half_size=[self.inner_width/2, self.inner_depth/2, self.wall_thickness/2]
        )
        drawer.add_box_visual(
            sapien.Pose([0, 0, -self.inner_height/2]),
            half_size=[self.inner_width/2, self.inner_depth/2, self.wall_thickness/2],
        )
        
        # Drawer right 
        drawer.add_box_collision(
            sapien.Pose([0, -self.inner_depth/2, 0]),
            half_size=[self.inner_width/2, self.wall_thickness/2, self.inner_height/2]
        )
        drawer.add_box_visual(
            sapien.Pose([0, -self.inner_depth/2, 0]),
            half_size=[self.inner_width/2, self.wall_thickness/2, self.inner_height/2],
        )
        
        # Drawer left 
        drawer.add_box_collision(
            sapien.Pose([0, self.inner_depth/2, 0]),
            half_size=[self.inner_width/2, self.wall_thickness/2, self.inner_height/2]
        )
        drawer.add_box_visual(
            sapien.Pose([0, self.inner_depth/2, 0]),
            half_size=[self.inner_width/2, self.wall_thickness/2, self.inner_height/2],
        )
        
        # Drawer back
        drawer.add_box_collision(
            sapien.Pose([self.inner_width/2, 0, 0]),
            half_size=[self.wall_thickness/2, self.inner_depth/2, self.inner_height/2]
        )
        drawer.add_box_visual(
            sapien.Pose([self.inner_width/2, 0, 0]),
            half_size=[self.wall_thickness/2, self.inner_depth/2, self.inner_height/2],
        )

        # Drawer front 
        drawer.add_box_collision(
            sapien.Pose([-self.inner_width/2, 0, 0]),
            half_size=[self.wall_thickness/2, self.inner_depth/2, self.inner_height/2]
        )
        drawer.add_box_visual(
            sapien.Pose([-self.inner_width/2, 0, 0]),
            half_size=[self.wall_thickness/2, self.inner_depth/2, self.inner_height/2],
        )
        
        # Handle material 
        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 0, 0, 1])
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0
        
        # Main handle bar 
        drawer.add_box_collision(
            sapien.Pose([-self.inner_width/2 - self.handle_offset, 0, 0]),
            half_size=[self.handle_thickness/2, self.handle_width/2, self.handle_thickness/2]
        )
        drawer.add_box_visual(
            sapien.Pose([-self.inner_width/2 - self.handle_offset, 0, 0]),
            half_size=[self.handle_thickness/2, self.handle_width/2, self.handle_thickness/2],
            material=mat
        )
        
        # Handle supports 
        for y_sign in [-1, 1]:
            support_y = y_sign * (self.handle_width/2 - self.handle_thickness/2)
            drawer.add_box_collision(
                sapien.Pose([-self.inner_width/2 - self.handle_offset/2, support_y, 0]),
                half_size=[self.handle_offset/2, self.handle_thickness/2, self.handle_thickness/2]
            )
            drawer.add_box_visual(
                sapien.Pose([-self.inner_width/2 - self.handle_offset/2, support_y, 0]),
                half_size=[self.handle_offset/2, self.handle_thickness/2, self.handle_thickness/2],
                material=mat
            )

        # Configure drawer joint 
        drawer.set_joint_properties(
            type="prismatic",
            limits=(-self.max_pull_distance, 0),
            pose_in_parent=sapien.Pose(),
            pose_in_child=sapien.Pose(),                
            friction=0.4,
            damping=10
        )

        builder.set_scene_idxs(scene_idxs=range(self.num_envs))
        # builder.set_initial_pose(sapien.Pose(p=[0.17, 0.15, 0.12]))  
          
        self.drawer = builder.build(fix_root_link=True, name="drawer_articulation")
        self.drawer_link = self.drawer.get_links()[1]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)
            
            drawer_xyz = torch.zeros((b, 3), device=self.device)
            drawer_xyz[..., 0] = torch.rand((b,), device=self.device) * self.k + 0.17
            drawer_xyz[..., 1] = torch.rand((b,), device=self.device) * self.k + 0.15
            drawer_xyz[..., 2] = self.outer_height / 2 + 0.005 


            init_pos = Pose.create_from_pq(p=drawer_xyz)
            
            self.drawer.set_pose(init_pos)

            closed_qpos = torch.zeros((b, 1), device=self.device)
            self.drawer.set_qpos(closed_qpos)

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
        drawer_pulled = pos_dist.squeeze(-1) < 0.03
        
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
        # Batch size extraction
        batch_size = self.drawer.get_qpos().shape[0]
        device = self.device

        self.scene._gpu_apply_all()
        self.scene.px.gpu_update_articulation_kinematics()
        self.scene._gpu_fetch_all()
        
        # Get TCP pose and drawer link pose
        tcp_pose = self.agent.tcp.pose.raw_pose
        tcp_pos = tcp_pose[..., :3]
        drawer_link_pose = self.drawer.links_map['drawer'].pose.raw_pose
        drawer_pose = self.drawer.pose.raw_pose

        handle_offset = torch.tensor([-self.inner_width/2 - self.handle_offset, 0, 0], device=drawer_link_pose.device)
        handle_pose = drawer_pose[:, :3] + handle_offset

        # 1. Orientation Reward - Modified for continuous feedback
        tcp_pose_q = self.agent.tcp.pose.q  # Current quaternion
        desired_q = torch.tensor([0.5, 0.5, 0.5, 0.5], device=device).expand(batch_size, 4)
        
        # Calculate quaternion distance (dot product between quaternions)
        # Abs because q and -q represent the same rotation
        quat_dot = torch.abs(torch.sum(tcp_pose_q * desired_q, dim=-1))
        # Clip to handle numerical errors
        quat_dot = torch.clamp(quat_dot, -1.0, 1.0)
        # Convert to angle (in radians)
        angle_dist = 2.0 * torch.acos(quat_dot)
        # Normalize to [0, 1] range and invert so smaller angles give higher rewards
        orientation_reward = 4.0 * (1.0 - torch.tanh(2.0 * angle_dist))

        # 2. Approach Reward
        reach_dist = torch.norm(tcp_pos - handle_pose, dim=-1)
        approach_reward = 4.0 * (1 - torch.tanh(5.0 * reach_dist))

        # 3. Progress Reward
        drawer_qpos = self.drawer.get_qpos()
        pos_dist = torch.abs(self.target_pos - drawer_qpos)
        pulling_reward = 4.0 * (1 - torch.tanh(5.0 * pos_dist)).squeeze(-1)

        # 4. Success Reward
        success_mask = info.get("success", torch.zeros_like(pulling_reward, dtype=torch.bool))
        completion_reward = 4.0 * success_mask
        
        return orientation_reward + approach_reward + pulling_reward + completion_reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 16.0  # Maximum possible reward
        dense_reward = self.compute_dense_reward(obs=obs, action=action, info=info)
        return dense_reward / max_reward
