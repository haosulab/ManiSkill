from typing import Union

import numpy as np
import torch

from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
import sapien
from mani_skill.envs.utils import randomization
from mani_skill.utils import common
from mani_skill.utils.building import actors
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.distraction_set import DistractionSet

from mani_skill.envs.tasks.tabletop.get_camera_config import get_camera_configs, get_human_render_camera_config

@register_env("StackCube-v2", max_episode_steps=50)
class StackCubeV2Env(StackCubeEnv):
    """
    Derived from StackCubeEnv, but with 3 cameras instead of 1. The dimensions of the cameras can be set as well.
    """
    def __init__(
        self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs
    ):
        assert "camera_width" in kwargs, "camera_width must be provided"
        assert "camera_height" in kwargs, "camera_height must be provided"
        self._camera_width = kwargs.pop("camera_width")
        self._camera_height = kwargs.pop("camera_height")
        self._human_render_shader = kwargs.pop("human_render_shader", None)
        # Distraction set
        self._distraction_set: Union[DistractionSet, dict] = kwargs.pop("distraction_set")
        if isinstance(self._distraction_set, dict):
            self._distraction_set = DistractionSet(**self._distraction_set)
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)


    def _load_scene(self, options: dict):

        self.cube_half_size = common.to_tensor([0.02] * 3, device=self.device)

        # === Add randomized tables
        self._table_scenes = []
        add_visual_from_file = not self._distraction_set.table_color_enabled()
        for i in range(self.num_envs):
            table_scene = TableSceneBuilder(self, robot_init_qpos_noise=self.robot_init_qpos_noise)
            table_scene.build(remove_table_from_state_dict_registry=True, scene_idx=i, name_suffix=f"env-{i}", add_visual_from_file=add_visual_from_file)
            self._table_scenes.append(table_scene)
        self.table_scene = Actor.merge([ts.table for ts in self._table_scenes], name="table_scene")
        self.add_to_state_dict_registry(self.table_scene)
        # ===


        # Create cube actors
        cubeA_actors = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=[0.02] * 3)
            builder.add_box_visual(
                half_size=[0.02] * 3,
                material=sapien.render.RenderMaterial(
                    base_color=[1, 0, 0, 1],
                ),
            )
            builder.set_scene_idxs([i])
            builder.initial_pose = sapien.Pose(p=[0, 0, 0.02])
            actor = builder.build_dynamic(name=f"cubeA_{i}")
            self.remove_from_state_dict_registry(actor)
            cubeA_actors.append(actor)
        self.cubeA = Actor.merge(cubeA_actors, name="cubeA")
        self.add_to_state_dict_registry(self.cubeA)

        cubeB_actors = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=[0.02] * 3)
            builder.add_box_visual(
                half_size=[0.02] * 3,
                material=sapien.render.RenderMaterial(
                    base_color=[0, 1, 0, 1],
                ),
            )
            builder.set_scene_idxs([i])
            builder.initial_pose = sapien.Pose(p=[0, 0, 0.02])
            actor = builder.build_dynamic(name=f"cubeB_{i}")
            self.remove_from_state_dict_registry(actor)
            cubeB_actors.append(actor)
        self.cubeB = Actor.merge(cubeB_actors, name="cubeB")
        self.add_to_state_dict_registry(self.cubeB)

        self._distraction_set.load_scene_hook(self.scene, manipulation_object=self.cubeA, table=self.table_scene)

        # self.cubeA = actors.build_cube(
        #     self.scene,
        #     half_size=0.02,
        #     color=[1, 0, 0, 1],
        #     name="cubeA",
        #     initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        # )
        # self.cubeB = actors.build_cube(
        #     self.scene,
        #     half_size=0.02,
        #     color=[0, 1, 0, 1],
        #     name="cubeB",
        #     initial_pose=sapien.Pose(p=[1, 0, 0.1]),
        # )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02
            xy = torch.rand((b, 2)) * 0.2 - 0.1
            region = [[-0.1, -0.2], [0.1, 0.2]]
            sampler = randomization.UniformPlacementSampler(
                bounds=region, batch_size=b, device=self.device
            )
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
            cubeA_xy = xy + sampler.sample(radius, 100)
            cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)

            xyz[:, :2] = cubeA_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = cubeB_xy
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeB.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            # Changed:
            # self.table_scene.initialize(env_idx)
            for ts in self._table_scenes:
                ts.initialize(env_idx)
            self._distraction_set.initialize_episode_hook(b, mo_pose=xyz)



    @property
    def _default_human_render_camera_configs(self):
        return get_human_render_camera_config(eye=[0.5, 0.2, 0.5], target=[-0.1, 0, 0.1], shader=self._human_render_shader)

    @property
    def _default_sensor_configs(self):
        target = [0, 0, 0.0]
        eye_xy = 0.3
        eye_z = 0.4
        cfgs = get_camera_configs(eye_xy, eye_z, target, self._camera_width, self._camera_height)
        cfgs_adjusted = self._distraction_set.update_camera_configs(cfgs)
        return cfgs_adjusted