import numpy as np
from typing import Union

import torch
import sapien

from mani_skill.utils.structs import Link, Actor
import mani_skill.envs.utils.randomization as randomization
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building import actors
from mani_skill.envs.distraction_set import DistractionSet

from mani_skill.envs.tasks.tabletop.get_camera_config import get_camera_configs

DEFAULT_GOAL_THRESH_MARGIN = 0.05

@register_env("PickCube-v2", max_episode_steps=100)
class PickCubeV2Env(PickCubeEnv):
    """
    **Task Description:**
    Nearly exacty copy of PickCubeEnv, but with the following change:
        1. 3 cameras instead of 1
        2. Cameras have a higher resolution
        3. Target position is fixed to (0.05, 0.05, 0.25)
        4. Goal_thresh is the cube half size plus a configurable margin
    """
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, goal_thresh_margin=DEFAULT_GOAL_THRESH_MARGIN, **kwargs):
        assert "camera_width" in kwargs, "camera_width must be provided"
        assert "camera_height" in kwargs, "camera_height must be provided"
        assert "distraction_set" in kwargs, "distraction_set must be provided"
        self._camera_width = kwargs.pop("camera_width")
        self._camera_height = kwargs.pop("camera_height")
        self._distraction_set: Union[DistractionSet, dict] = kwargs.pop("distraction_set")
        self._goal_thresh_margin = goal_thresh_margin
        # In this situation, the DistractionSet has serialized as a dict so we now need to deserialize it.
        if isinstance(self._distraction_set, dict):
            self._distraction_set = DistractionSet(**self._distraction_set)

        # Env configuration
        self.cube_half_size = 0.02
        self.goal_thresh = self.cube_half_size + self._goal_thresh_margin

        self._table_scenes: list[TableSceneBuilder] = []

        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)
        print(" --> Created PickCubeV2Env")


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        print("  PickCubeV2Env: _initialize_episode()")
        with torch.device(self.device):
            b = len(env_idx) # number of environments. env_idx is [0, 1, 2, ..., n_envs-1]
            for ts in self._table_scenes:
                ts.initialize(env_idx)

            # Random cube position
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            # Fixed target position
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, 0] = 0.05
            goal_xyz[:, 1] = 0.05
            goal_xyz[:, 2] = 0.25
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

            # Initialize distraction set
            self._distraction_set.initialize_episode_hook(b, xyz)


    def _load_scene(self, options: dict):
        """ Load the scene.
        """
        print("  PickCubeV2Env: _load_scene()")
        self._table_scenes = []
        add_visual_from_file = not self._distraction_set.table_color_enabled()
        for i in range(self.num_envs):
            table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
            table_scene.build(remove_table_from_state_dict_registry=True, scene_idx=i, name_suffix=f"env-{i}", add_visual_from_file=add_visual_from_file)
            self._table_scenes.append(table_scene)
        self.table_scene = Actor.merge([ts.table for ts in self._table_scenes], name="table_scene")
        self.add_to_state_dict_registry(self.table_scene)


        # Create cube actors
        cube_actors = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=[self.cube_half_size] * 3)
            builder.add_box_visual(
                half_size=[self.cube_half_size] * 3,
                material=sapien.render.RenderMaterial(
                    base_color=[1, 0, 0, 1],
                ),
            )
            builder.set_scene_idxs([i])
            builder.initial_pose = sapien.Pose(p=[0, 0, self.cube_half_size])
            actor = builder.build_dynamic(name=f"cube_{i}")
            self.remove_from_state_dict_registry(actor)
            cube_actors.append(actor)
        self.cube = Actor.merge(cube_actors, name="cube")
        self.add_to_state_dict_registry(self.cube)


        # Create goal site
        print(f"PickCubeV2Env: Creating goal site with radius: {self.goal_thresh}")
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 0.75],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)
        self._distraction_set.load_scene_hook(self.scene, manipulation_object=self.cube, table=self.table_scene)


    @property
    def _default_human_render_camera_configs(self):
        """ Configures the human render camera.

        Shader options:
            minimal: The fastest shader with minimal GPU memory usage. Note that the background will always be black (normally it is the color of the ambient light)
            default: A balance between speed and texture availability
            rt: A shader optimized for photo-realistic rendering via ray-tracing
            rt-med: Same as rt but runs faster with slightly lower quality
            rt-fast: Same as rt-med but runs faster with slightly lower quality

            -> https://maniskill.readthedocs.io/en/latest/user_guide/concepts/sensors.html#shaders-and-textures
        """
        print("  PickCubeV2Env: _default_human_render_camera_configs()")
        pose = sapien_utils.look_at([0.35, 0.45, 0.4], [0.0, 0.0, 0.15])
        SHADER = "default"
        return CameraConfig("render_camera", pose=pose, width=1264, height=1264, fov=np.pi / 3, near=0.01, far=100, shader_pack=SHADER)

    @property
    def _default_sensor_configs(self):
        print("  PickCubeV2Env: _default_sensor_configs()")
        target=[-0.1, 0, 0.1]
        eye_xy = 0.3
        eye_z = 0.6
        cfgs = get_camera_configs(eye_xy, eye_z, target, self._camera_width, self._camera_height)
        cfgs_adjusted = self._distraction_set.update_camera_configs(cfgs)
        return cfgs_adjusted