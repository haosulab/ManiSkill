import numpy as np
from typing import Union

import torch
import sapien

from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Fetch, Panda, XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs import Actor
import mani_skill.envs.utils.randomization as randomization
from mani_skill.envs.distraction_set import DistractionSet
from mani_skill.envs.tasks.tabletop.get_camera_config import get_camera_configs, get_human_render_camera_config

DEFAULT_GOAL_THRESH_MARGIN = 0.05

@register_env("RaiseCube-v1", max_episode_steps=100)
class RaiseCubeEnv(BaseEnv):

    SUPPORTED_ROBOTS = [
        "panda",
        "panda_wristcam",
        "fetch",
        "xarm6_robotiq",
    ]
    agent: Union[Panda, Fetch, XArm6Robotiq]
    cube_half_size = 0.02
    target_height = 0.15

    """
    """
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        assert "camera_width" in kwargs, "camera_width must be provided"
        assert "camera_height" in kwargs, "camera_height must be provided"
        assert "distraction_set" in kwargs, "distraction_set must be provided"
        self._camera_width = kwargs.pop("camera_width")
        self._camera_height = kwargs.pop("camera_height")
        self._distraction_set: Union[DistractionSet, dict] = kwargs.pop("distraction_set")
        if isinstance(self._distraction_set, dict):
            # In this situation, the DistractionSet has serialized as a dict so we now need to deserialize it.
            self._distraction_set = DistractionSet(**self._distraction_set)
        self._human_render_shader = kwargs.pop("human_render_shader", None)
        # Env configuration
        self.cube_half_size = 0.02
        self._table_scenes: list[TableSceneBuilder] = []
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)


    def evaluate(self):
        is_obj_raised = self.cube.pose.p[..., 2] > self.target_height
        is_robot_static = self.agent.is_static(0.2)

        # 
        return {
            "success": is_obj_raised & is_robot_static,
            "is_obj_raised": is_obj_raised,
            "is_robot_static": is_robot_static,
        }


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
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

            # Initialize distraction set
            self._distraction_set.initialize_episode_hook(n_envs=b, mo_pose=xyz, ro_pose=None)


    def _load_scene(self, options: dict):
        """ Load the scene.
        """
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


        self._distraction_set.load_scene_hook(self.scene, manipulation_object=self.cube, table=self.table_scene, receiving_object=None)


    @property
    def _default_human_render_camera_configs(self):
        return get_human_render_camera_config(eye=[0.35, 0.45, 0.4], target=[0.0, 0.0, 0.15], shader=self._human_render_shader)

    @property
    def _default_sensor_configs(self):
        target=[0.0, 0, 0.15]
        xy_offset = 0.3
        z_offset = 0.4
        cfgs = get_camera_configs(xy_offset, z_offset, target, self._camera_width, self._camera_height)
        cfgs_adjusted = self._distraction_set.update_camera_configs(cfgs)
        return cfgs_adjusted

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

