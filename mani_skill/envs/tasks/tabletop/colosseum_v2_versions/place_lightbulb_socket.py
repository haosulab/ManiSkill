import numpy as np
import torch
import sapien
from typing import Dict, Any, Union
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building import actors
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.agents.robots import Panda, Fetch
import os
from mani_skill.envs.distraction_set import DistractionSet


@register_env("PickLightbulbPlaceSocket-v1", max_episode_steps=100000)
class PickLightbulbPlaceSocketEnv(BaseEnv):
    """Pick up a lightbulb and place it into a lamp socket."""
    
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]
    
    socket_tolerance = 0.03
    alignment_tolerance = 0.15
    
    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.8, 0.8, 0.8], target=[0.0, 0.0, 0.35])
        return [
            CameraConfig(
                "viewer_cam",
                pose=pose,
                width=960,
                height=720,
                fov=np.pi / 3,
                near=0.01,
                far=100.0,
            )
        ]
    
    def __init__(self, *args, robot_uids="panda", num_envs=1, 
                 reconfiguration_freq=None, **kwargs):
        distraction_set: Union[DistractionSet, dict] = kwargs.pop("distraction_set")
        self._distraction_set: DistractionSet = DistractionSet(**distraction_set) if isinstance(distraction_set, dict) else distraction_set
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0
        super().__init__(*args, robot_uids=robot_uids, 
                        reconfiguration_freq=reconfiguration_freq, 
                        num_envs=num_envs, **kwargs)
    
    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))
    
    @staticmethod
    def load_glb_as_actor(scene, glb_file_path, pose, name, body_type="dynamic"):
        builder = scene.create_actor_builder()
        builder.add_visual_from_file(glb_file_path)
        builder.add_multiple_convex_collisions_from_file(
            glb_file_path, 
            decomposition="coacd"
        )
        builder.initial_pose = pose
        
        if body_type == "dynamic":
            actor = builder.build(name=name)
        elif body_type == "kinematic":
            actor = builder.build_kinematic(name=name)
        else:
            actor = builder.build_static(name=name)
        
        return actor
    
    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()
        
        lightbulb_path = "/mnt/user-data/uploads/Lightbulb.glb"
        if os.path.exists(lightbulb_path):
            self.lightbulb = self.load_glb_as_actor(
                self.scene,
                lightbulb_path,
                sapien.Pose(p=[0.2, 0, 0.1]),
                "lightbulb",
                body_type="dynamic"
            )
            self.lightbulb.set_mass(0.05)
            print("Loaded lightbulb from GLB file")
        else:
            print("GLB not found, using fallback bulb shape")
            self._create_fallback_lightbulb()
        
        self._create_lamp_socket()
    
    def _create_fallback_lightbulb(self):
        self.lightbulb = actors.build_sphere(
            self.scene,
            radius=0.03,
            color=[1, 1, 0.8, 1],
            name="lightbulb",
            initial_pose=sapien.Pose(p=[0.2, 0, 0.1])
        )
        self.lightbulb.set_mass(0.05)
    
    def _create_lamp_socket(self):
        self.lamp_base = actors.build_cylinder(
            self.scene,
            radius=0.08,
            half_length=0.01,
            color=[0.2, 0.2, 0.2, 1],
            name="lamp_base",
            body_type="static",
            initial_pose=sapien.Pose(p=[0.35, 0.15, 0.01])
        )
        
        self.lamp_pole = actors.build_cylinder(
            self.scene,
            radius=0.01,
            half_length=0.15,
            color=[0.3, 0.3, 0.3, 1],
            name="lamp_pole", 
            body_type="static",
            initial_pose=sapien.Pose(p=[0.35, 0.15, 0.16])
        )
        
        socket_builder = self.scene.create_actor_builder()
        socket_builder.add_cylinder_visual(
            radius=0.018,
            half_length=0.02
        )
        socket_builder.initial_pose = sapien.Pose(p=[0.35, 0.15, 0.28])
        self.lamp_socket = socket_builder.build_static(name="lamp_socket")
        
        self.socket_marker = actors.build_sphere(
            self.scene,
            radius=0.015,
            color=[0, 1, 0, 0.3],
            name="socket_marker",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0.35, 0.15, 0.28])
        )
        
        self.socket_pos = np.array([0.35, 0.15, 0.28])
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            
            self.table_scene.initialize(env_idx)
            
            bulb_xy = torch.rand((b, 2)) * 0.2 - 0.1
            bulb_xy[:, 0] += 0.2
            bulb_z = torch.full((b, 1), 0.05)
            bulb_p = torch.cat([bulb_xy, bulb_z], dim=-1)
            
            q = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32).expand(b, -1)
            
            self.lightbulb.set_pose(Pose.create_from_pq(p=bulb_p, q=q))
            
            self.socket_pos_tensor = torch.tensor(
                [[self.socket_pos[0], self.socket_pos[1], self.socket_pos[2]]], 
                dtype=torch.float32
            ).expand(b, -1)
    
    def evaluate(self):
        bulb_pos = self.lightbulb.pose.p
        socket_pos = self.socket_pos_tensor
        
        xy_dist = torch.linalg.norm(bulb_pos[:, :2] - socket_pos[:, :2], dim=1)
        xy_aligned = xy_dist < self.socket_tolerance
        
        z_diff = torch.abs(bulb_pos[:, 2] - socket_pos[:, 2])
        z_aligned = z_diff < 0.02
        
        bulb_q = self.lightbulb.pose.q
        upright = torch.abs(bulb_q[:, 0]) > 0.9
        
        is_static = self.lightbulb.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        
        success = xy_aligned & z_aligned & upright & is_static
        
        return {
            "success": success,
            "xy_aligned": xy_aligned,
            "z_aligned": z_aligned,
            "upright": upright,
            "is_static": is_static,
            "xy_dist": xy_dist,
            "z_diff": z_diff,
        }
    
    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                lightbulb_pose=self.lightbulb.pose.raw_pose,
                socket_position=self.socket_pos_tensor,
            )
        return obs
    
    def compute_normalized_dense_reward(self, obs: Any, action: np.ndarray, info: Dict):
        xy_dist = info["xy_dist"]
        z_diff = info["z_diff"]
        
        dist_reward = 1.0 / (1.0 + xy_dist * 10.0)
        height_reward = 1.0 / (1.0 + z_diff * 10.0)
        
        upright_bonus = info["upright"].float() * 0.5
        
        success_bonus = info["success"].float() * 3.0
        
        reward = dist_reward + height_reward + upright_bonus + success_bonus
        return torch.clamp(reward / 6.0, -1.0, 1.0)