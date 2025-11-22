from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils.structs import Actor
from mani_skill.agents.robots import SO100, Fetch, Panda, WidowXAI, XArm6Robotiq
from mani_skill.envs.distraction_set import DistractionSet
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.envs.tasks.tabletop.colosseum_v2_versions.colosseum_v2_env_utils import get_camera_configs, get_human_render_camera_config

PICK_CUBE_DOC_STRING = """**Task Description:**
A simple task where the objective is to grasp a red cube with the {robot_id} robot and move it to a target goal position. This is also the *baseline* task to test whether a robot with manipulation
capabilities can be simulated and trained properly. Hence there is extra code for some robots to set them up properly in this environment as well as the table scene builder.

**Randomizations:**
- the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
- the cube's z-axis rotation is randomized to a random angle
- the target goal position (marked by a green sphere) of the cube has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

**Success Conditions:**
- the cube position is within `goal_thresh` (default 0.025m) euclidean distance of the goal position
- the robot is static (q velocity < 0.2)
"""


@register_env("RaiseCube-v1", max_episode_steps=50)
class RaiseCubeEnv(BaseEnv):

    """ This is a copy of the PickCube-v1 environment, but rather than reaching a goal position after grasping, the 
    cube simply needs to be raised above a target height.
    """

    GOAL_HEIGHT = 0.2

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):

        # ColosseumV2 stuff:
        distraction_set: Union[DistractionSet, dict] = kwargs.pop("distraction_set")
        self._distraction_set: DistractionSet = DistractionSet(**distraction_set) if isinstance(distraction_set, dict) else distraction_set
        self._human_render_shader = kwargs.pop("human_render_shader", None)


        # 
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot_uids in PICK_CUBE_CONFIGS:
            cfg = PICK_CUBE_CONFIGS[robot_uids]
        else:
            cfg = PICK_CUBE_CONFIGS["panda"]
        self.cube_half_size = cfg["cube_half_size"]
        self.goal_thresh = cfg["goal_thresh"]
        self.cube_spawn_half_size = cfg["cube_spawn_half_size"]
        self.cube_spawn_center = cfg["cube_spawn_center"]
        self.max_goal_height = cfg["max_goal_height"]
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]
        super().__init__(*args, robot_uids=robot_uids, **kwargs)


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
        self.cube = Actor.merge(cube_actors, name="cube")
        self.add_to_state_dict_registry(self.cube)

        # load_scene_hook(self, scene: ManiSkillScene, manipulation_object: Optional[Actor], table: Optional[Actor], receiving_object: Optional[Actor])
        self._distraction_set.load_scene_hook(scene=self.scene, manipulation_object=self.cube, table=self.table_scene)


    @property
    def _default_human_render_camera_configs(self):
        return get_human_render_camera_config(eye=(0.5, 0.6, 0.5), target=(0.0, 0.0, 0.1), shader=self._human_render_shader)


    @property
    def _default_sensor_configs(self):
        target=(0.1, 0, -0.1)
        eye_xy = 0.35
        eye_z = 0.45
        cfgs = get_camera_configs(eye_xy, eye_z, target)
        cfgs_adjusted = self._distraction_set.update_camera_configs(cfgs)
        return cfgs_adjusted


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
            builder.add_box_collision(half_size=(self.cube_half_size, self.cube_half_size, self.cube_half_size))
            builder.add_box_visual(
                half_size=(self.cube_half_size, self.cube_half_size, self.cube_half_size),
                material=sapien.render.RenderMaterial(
                    base_color=np.array([12, 42, 160, 255]) / 255,
                ),
            )
            builder.set_scene_idxs([i])
            builder.initial_pose = sapien.Pose(p=[0, 0, self.cube_half_size])
            actor = builder.build_dynamic(name=f"cube_{i}")
            self.remove_from_state_dict_registry(actor)
            cube_actors.append(actor)
        self.cube = Actor.merge(cube_actors, name="cube")
        self.add_to_state_dict_registry(self.cube)

        # load_scene_hook(self, scene: ManiSkillScene, manipulation_object: Optional[Actor], table: Optional[Actor], receiving_object: Optional[Actor])
        self._distraction_set.load_scene_hook(scene=self.scene, manipulation_object=self.cube, table=self.table_scene)



    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            for ts in self._table_scenes:
                ts.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[..., 2] = self.cube_half_size
            q = [1, 0, 0, 0]

            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.cube.set_pose(obj_pose)

            # 
            self._distraction_set.initialize_episode_hook(b, mo_pose=xyz)
        
    def _get_obs_extra(self, info: dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp_pose.p,
            )
        return obs

    def evaluate(self):
        is_obj_raised = self.cube.pose.p[..., 2] > self.GOAL_HEIGHT
        is_grasped = self.agent.is_grasping(self.cube)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_raised & is_robot_static,
            "is_obj_raised": is_obj_raised,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

