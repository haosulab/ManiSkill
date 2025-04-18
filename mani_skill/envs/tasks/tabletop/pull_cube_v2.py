from typing import Union

import numpy as np
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.utils.structs import Actor
from mani_skill.envs.tasks.tabletop.pull_cube import PullCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building import actors
from mani_skill.envs.distraction_set import DistractionSet

from mani_skill.envs.tasks.tabletop.get_camera_config import get_camera_configs

@register_env("PullCube-v2", max_episode_steps=50)
class PullCubeV2Env(PullCubeEnv):
    """
    **Task Description:**
    Nearly exacty copy of PullCubeEnv, but with 3 cameras instead of 1.

    Notes:
     - Cube spawns in x: [-0.1, 0.1], y: [-0.1, 0.1]
     - Goal region is cube-position - [0.1 + goal_radius (=0.1), 0, 0]

     -> min workspace size should be x: [-0.3, 0.1], y: [-0.1, 0.1]
     -> x-mid = -0.1, y-mid = 0.0
    """
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self._camera_width = kwargs.pop("camera_width")
        self._camera_height = kwargs.pop("camera_height")
        self._distraction_set: Union[DistractionSet, dict] = kwargs.pop("distraction_set")
        # In this situation, the DistractionSet has serialized as a dict so we now need to deserialize it.
        if isinstance(self._distraction_set, dict):
            self._distraction_set = DistractionSet(**self._distraction_set)

        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            for ts in self._table_scenes:
                ts.initialize(env_idx)

            # self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[..., 2] = self.cube_half_size
            q = [1, 0, 0, 0]

            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.obj.set_pose(obj_pose)

            target_region_xyz = xyz - torch.tensor([0.1 + self.goal_radius, 0, 0])
            target_region_xyz[..., 2] = 1e-3
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=target_region_xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )
            self._distraction_set.initialize_episode_hook(b, mo_pose=xyz)


    def _load_scene(self, options: dict):

        # Create table
        self._table_scenes = []
        add_visual_from_file = not self._distraction_set.table_color_enabled()
        for i in range(self.num_envs):
            table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
            table_scene.build(remove_table_from_state_dict_registry=True, scene_idx=i, name_suffix=f"-env-{i}", add_visual_from_file=add_visual_from_file)
            self._table_scenes.append(table_scene)
        self.table_scene = Actor.merge([ts.table for ts in self._table_scenes], name="table")
        self.add_to_state_dict_registry(self.table_scene)

        # Create cube
        cube_actors = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=[self.cube_half_size] * 3)
            builder.add_box_visual(
                half_size=[self.cube_half_size] * 3,
                material=sapien.render.RenderMaterial(
                    base_color=np.array([12, 42, 160, 255]) / 255,
                ),
            )
            builder.set_scene_idxs([i])
            builder.initial_pose = sapien.Pose(p=[0, 0, self.cube_half_size])
            actor = builder.build_dynamic(name=f"cube_{i}")
            self.remove_from_state_dict_registry(actor)
            cube_actors.append(actor)
        self.obj = Actor.merge(cube_actors, name="cube")
        self.add_to_state_dict_registry(self.obj)

        # create target
        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose()
        )
        self._distraction_set.load_scene_hook(self.scene, manipulation_object=self.obj, table=self.table_scene)



    @property
    def _default_human_render_camera_configs(self):
        """ Configures the human render camera.
        """
        pose = sapien_utils.look_at([0.5, 0.6, 0.5], [0.0, 0.0, 0.1])
        SHADER = "default"
        return CameraConfig("render_camera", pose=pose, width=1264, height=1264, fov=np.pi / 3, near=0.01, far=100, shader_pack=SHADER)


    @property
    def _default_sensor_configs(self):
        SHADER = "default"
        target=[-0.1, 0, -0.1]
        eye_xy = 0.35
        eye_z = 0.45
        cfgs = get_camera_configs(eye_xy, eye_z, target, self._camera_width, self._camera_height)
        cfgs_adjusted = self._distraction_set.update_camera_configs(cfgs)
        return cfgs_adjusted