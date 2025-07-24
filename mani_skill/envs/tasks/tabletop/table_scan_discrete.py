# table_scan_env.py
from typing import Dict, Any, Union, List
import numpy as np, torch, sapien
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs import Pose, GPUMemoryConfig, SimConfig
from mani_skill.utils.building import actors
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env
from mani_skill.agents.robots import XArm6Robotiq
from mani_skill.utils.structs import Actor

@register_env("TableScanDiscreteInit-v0", max_episode_steps=1_000)
class TableScanDiscreteInitEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["xarm6_robotiq"]
    agent: XArm6Robotiq
        
    cube_half_size = 0.02
    cube_spawn_half_size = 0.1
    cube_spawn_center = (0, 0)
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
        
    def _load_lighting(self, options: dict):
        for scene in self.scene.sub_scenes:
            scene.ambient_light = [np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6), np.random.uniform(0.2, 0.6)]
            scene.add_directional_light(np.random.uniform(-1, 1, 3), [1, 1, 1], shadow=True, shadow_scale=5, shadow_map_size=4096)
            scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _load_scene(self, options: Dict):
        self.table_scene = TableSceneBuilder(
            env=self, custom_table=True, randomize_colors=True
        )
        self.table_scene.build()
        
        ### Cube randomization: Build cubes separately for each parallel environment to enable domain randomization        
        self._cubes: List[Actor] = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=[self.cube_half_size] * 3)
            builder.add_box_visual(
                half_size=[self.cube_half_size] * 3, 
                material=sapien.render.RenderMaterial(
                    base_color=self._batched_episode_rng[i].uniform(low=0., high=1., size=(3, )).tolist() + [1]
                )
            )
            builder.initial_pose = sapien.Pose(p=[0, 0, self.cube_half_size])
            builder.set_scene_idxs([i])
            self._cubes.append(builder.build(name=f"cube_{i}"))
            self.remove_from_state_dict_registry(self._cubes[-1])  # remove individual cube from state dict

        # Merge all cubes into a single Actor object
        self.cube = Actor.merge(self._cubes, name="cube")
        self.add_to_state_dict_registry(self.cube)  # add merged cube to state dict
        
        self.table_center = [0, 0, self.table_scene.table_height/2]
        self.cam_mount = self.scene.create_actor_builder().build_kinematic("camera_mount")
        
        self._theta_min  = self.THETA_MIN
        self._theta_max  = self.THETA_MAX
        self._cam_theta  = self._theta_min         
        self._cam_speed  = self.CAM_SPEED

        self._cam_idx    = 0                       
        self._cam_height = self.VIEW_HEIGHTS[self._cam_idx]
        self._cam_radius = self.CAM_RADIUS
        
    def _reconfigure(self, options=dict()):
        """Clean up individual actors created for domain randomization to prevent memory leaks during resets."""
        if hasattr(self, '_cubes'):
            # Remove individual cubes from the scene
            for cube in self._cubes:
                if hasattr(cube, 'entity') and cube.entity is not None:
                    self.scene.remove_actor(cube)
            self._cubes.clear()
        
        # Clean up table scene builder if it exists
        if hasattr(self, 'table_scene'):
            self.table_scene.cleanup()

        super()._reconfigure(options)

    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            
            xyz = torch.zeros((b, 3))
            
            # Cube position chosen from a 10x10 grid
            grid_idx = env_idx % 100
            x_grid = grid_idx // 10
            y_grid = grid_idx % 10
            xyz[:, 0] = torch.linspace(0, 1, 10)[x_grid] * self.cube_spawn_half_size * 2 - self.cube_spawn_half_size
            xyz[:, 1] = torch.linspace(0, 1, 10)[y_grid] * self.cube_spawn_half_size * 2 - self.cube_spawn_half_size
            xyz[:, 0] += self.cube_spawn_center[0]
            xyz[:, 1] += self.cube_spawn_center[1]
            xyz[:, 2] = self.cube_half_size
 
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

    def evaluate(self):
        return {"success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)}

    def compute_dense_reward(self, *_, **__):
        return torch.zeros(self.num_envs, device=self.device)

    def get_table_center(self):
        return self.table_center
