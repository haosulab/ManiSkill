# table_scan_env.py
from typing import Dict, Any, Union, List
import numpy as np, torch, sapien
from transforms3d.euler import euler2quat
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs import Pose, GPUMemoryConfig, SimConfig
from mani_skill.utils.building import actors
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env
from mani_skill.agents.robots import XArm6Robotiq

@register_env("TableScan-v0", max_episode_steps=1_000)
class TableScanEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["xarm6_robotiq"]
    agent: XArm6Robotiq
        
    CAM_RADIUS = 0.3
    VIEW_HEIGHTS = [0.35, 0.42, 0.55, 0.7]
    CAM_SPEED  = np.deg2rad(2) # rad / sim-step (≈115 °/s at 60 Hz)

    THETA_MIN    = np.deg2rad(-90)
    THETA_MAX    = np.deg2rad( 90)

    # Specify default simulation/gpu memory configurations to override any default values
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        """Hand-mounted camera 25 cm in front of the TCP, looking forward."""
        offset = [0.15, 0.0, -0.2]                     
        down90 = euler2quat(0,  -np.pi/2, 0)        
        hand_cam_pose = sapien.Pose(offset, down90)
        real_pose = sapien_utils.look_at(eye=[0.508, -0.5, 0.42], target=[-0.522, 0.2, 0])
        moving_camera = CameraConfig(
                            "moving_camera", pose=sapien.Pose(), width=224, height=224, 
                            fov=np.pi * 0.4, near=0.01, far=100, 
                            mount=self.cam_mount
                        )

        return [
            # CameraConfig(
            #     "hand_cam",
            #     pose=hand_cam_pose,
            #     width=128,
            #     height=128,
            #     fov=np.pi * 0.4,
            #     near=0.01,
            #     far=100,
            #     mount=self.agent.tcp,
            # )
            # CameraConfig(
            #     "hand_cam",
            #     pose=real_pose,
            #     # width=640,
            #     # height=480,
            #     width=128,
            #     height=128,
            #     fov=np.pi * 0.4,
            #     near=0.01,
            #     far=100,
            # )
            moving_camera
        ]
        
    def _before_simulation_step(self):
        super()._before_simulation_step()

        pos = self.table_center + np.array([
            self._cam_radius * np.cos(self._cam_theta),
            self._cam_radius * np.sin(self._cam_theta),
            self._cam_height,
        ])
        self.cam_mount.set_pose(sapien_utils.look_at(pos, self.table_center))

        self._cam_theta += self._cam_speed

        if self._cam_theta > self._theta_max or self._cam_theta < self._theta_min:
            self._cam_theta = np.clip(self._cam_theta, self._theta_min, self._theta_max)
            self._cam_speed *= -1

            self._cam_idx   = (self._cam_idx + 1) % len(self.VIEW_HEIGHTS)
            self._cam_height = self.VIEW_HEIGHTS[self._cam_idx]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )
    @property
    def _supports_sensor_data(self):
        return True

    def _load_agent(self, options: Dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: Dict):
        self.table_scene = TableSceneBuilder(
            env=self, custom_table=True
        )
        self.table_scene.build()
        self.table_center = [0, 0, self.table_scene.table_height/2]
        self.cam_mount = self.scene.create_actor_builder().build_kinematic("camera_mount")
        
        self._theta_min  = self.THETA_MIN
        self._theta_max  = self.THETA_MAX
        self._cam_theta  = self._theta_min         
        self._cam_speed  = self.CAM_SPEED

        self._cam_idx    = 0                       
        self._cam_height = self.VIEW_HEIGHTS[self._cam_idx]
        self._cam_radius = self.CAM_RADIUS

        rng = np.random.default_rng(0)
        for i in range(6):
            x, y = rng.uniform(-0.2, 0.2, 2)
            z = 0.03
            color = rng.uniform(0.2, 0.9, 3)
            actors.build_cube(
                self.scene,
                half_size=0.03,
                color=np.concatenate([color, [1.0]]),
                name=f"obj_{i}",
                body_type="dynamic",
                initial_pose=sapien.Pose([x, y, z]),
            )

    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

    def evaluate(self):
        return {"success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)}

    def compute_dense_reward(self, *_, **__):
        return torch.zeros(self.num_envs, device=self.device)

    def get_table_center(self):
        return self.table_center
