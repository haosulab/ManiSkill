from typing import Dict, Any, List
import numpy as np, torch, sapien
import gymnasium as gym
from transforms3d.quaternions import mat2quat
import transforms3d as t3d
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.utils.structs import Pose, GPUMemoryConfig, SimConfig
from mani_skill.utils.building import actors
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env


@register_env("TableScanDiscreteNoRobot-v0", max_episode_steps=1_000)

class TableScanDiscreteNoRobotEnv(BaseEnv):

    # Constants
    cube_half_size = 0.02
    cube_spawn_half_size = 0.10
    cube_spawn_center = (0, 0)

    def __init__(self, *args, **kwargs):

        self._traj_idx = 0
        self.camera_mount_offset = 0
        super().__init__(*args, **kwargs)

    # Specify default simulation/gpu memory configurations to override any default values
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18)
        )

    @property
    def _default_human_render_camera_configs(self):
        moving_camera = CameraConfig(
            "moving_camera", pose=sapien.Pose(), width=224, height=224,
            fov=np.pi * 0.4, near=0.01, far=100, mount=self.cam_mount
        )
        fixed_cam_pose = sapien_utils.look_at(eye=[0.508, -0.5, 0.42], target=[-0.522, 0.2, 0])
        return [
            CameraConfig("hand_cam", pose=fixed_cam_pose, width=224, height=224, fov=np.pi/2, near=0.01, far=100),
            moving_camera
        ]

    def _load_agent(self, options: Dict):
        self.agent = None

    def _load_scene(self, options: Dict):
        self.table_scene = TableSceneBuilder(env=self, custom_table=True, randomize_colors=False)
        self.table_scene.build()
        self.table_center = [0, 0, self.table_scene.table_height/2]
        self.cam_mount = self.scene.create_actor_builder().build_kinematic("camera_mount")

        # MARK: We first comment out the domain randomization part
        # ### Cube randomization: Build cubes separately for each parallel environment to enable domain randomization        
        # self._cubes: List[Actor] = []
        # for i in range(self.num_envs):
        #     builder = self.scene.create_actor_builder()
        #     builder.add_box_collision(half_size=[self.cube_half_size] * 3)
        #     builder.add_box_visual(
        #         half_size=[self.cube_half_size] * 3, 
        #         material=sapien.render.RenderMaterial(
        #             base_color=self._batched_episode_rng[i].uniform(low=0., high=1., size=(3, )).tolist() + [1]
        #         )
        #     )
        #     builder.initial_pose = sapien.Pose(p=[0, 0, self.cube_half_size])
        #     builder.set_scene_idxs([i])
        #     self._cubes.append(builder.build(name=f"cube_{i}"))
        #     self.remove_from_state_dict_registry(self._cubes[-1])  # remove individual cube from state dict

        # # Merge all cubes into a single Actor object
        # self.cube = Actor.merge(self._cubes, name="cube")
        # self.add_to_state_dict_registry(self.cube)  # add merged cube to state dict
        
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )

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
        self.step_count = 0
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

    def render_at_pose(self, pose: sapien.Pose = None) -> dict:
        camera = self.scene.human_render_cameras["moving_camera"]

        self.cam_mount.set_pose(pose)
        # self.scene.step()
        self.scene.update_render()  # update scene: object states, camera, etc.
        camera.camera.take_picture()  # start the rendering process
        obs = {k: v[0] for k, v in camera.get_obs(position=False).items()}
        if (
            "position" in obs
        ):
            obs["position"][..., 1] *= -1
            obs["position"][..., 2] *= -1
        obs["cam_pose"] = np.concatenate([pose.p, pose.q])
        obs["extrinsic_cv"] = camera.camera.get_extrinsic_matrix()[0]
        obs["intrinsic_cv"] = camera.camera.get_intrinsic_matrix()[0]
        return obs

    def _step_action(self, action):
        return action

    def _get_obs_agent(self):
        return {}

    def evaluate(self): return {"success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)}
    def compute_dense_reward(self, *_, **__): return torch.zeros(self.num_envs, device=self.device)
    def get_state_dict(self): return self.scene.get_sim_state()
    def set_state_dict(self, state): return self.scene.set_sim_state(state)