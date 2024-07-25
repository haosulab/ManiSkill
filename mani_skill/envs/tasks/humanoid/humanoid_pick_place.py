import copy
import os
from typing import Dict, Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.agents.robots.unitree_g1.g1_upper_body import UnitreeG1UpperBody
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.kitchen_counter import KitchenCounterSceneBuilder
from mani_skill.utils.structs.types import SimConfig


class HumanoidPickPlaceEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["sparse", "none"]
    """sets up a basic scene with a apple to pick up and place on a dish"""
    kitchen_scene_scale = 1.0

    def __init__(self, *args, robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1)

    def _load_scene(self, options: dict):
        self.scene_builder = KitchenCounterSceneBuilder(self)
        self.kitchen_scene = self.scene_builder.build(scale=self.kitchen_scene_scale)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        pass

    def evaluate(self):
        return {
            "success": torch.zeros(self.num_envs, device=self.device, dtype=bool),
            "fail": torch.zeros(self.num_envs, device=self.device, dtype=bool),
        }

    def _get_obs_extra(self, info: Dict):
        return dict()


class HumanoidPlaceAppleInBowl(HumanoidPickPlaceEnv):
    def _load_scene(self, options: Dict):
        super()._load_scene(options)
        scale = self.kitchen_scene_scale
        builder = self.scene.create_actor_builder()
        table_pose = sapien.Pose(q=euler2quat(np.pi / 2, 0, np.pi))
        model_dir = os.path.dirname(__file__) + "/assets"
        builder.add_multiple_convex_collisions_from_file(
            filename=os.path.join(model_dir, "frl_apartment_bowl_07.ply"),
            pose=table_pose,
            scale=[scale] * 3,
        )
        builder.add_visual_from_file(
            filename=os.path.join(model_dir, "frl_apartment_bowl_07.glb"),
            scale=[scale] * 3,
            pose=table_pose,
        )
        builder.initial_pose = sapien.Pose(p=[0, -0.4, 0.753])
        self.bowl = builder.build(name="bowl")


@register_env("UnitreeG1PlaceAppleInBowl-v1", max_episode_steps=100)
class UnitreeG1PlaceAppleInBowlEnv(HumanoidPlaceAppleInBowl):
    SUPPORTED_ROBOTS = ["unitree_g1_simplified_upper_body_right_arm"]
    agent: Union[UnitreeG1UpperBody]
    kitchen_scene_scale = 0.82

    def __init__(self, *args, **kwargs):
        self.init_robot_pose = copy.deepcopy(
            UnitreeG1UpperBody.keyframes["standing"].pose
        )
        self.init_robot_pose.p = [-0.35, 0, 0.755]
        super().__init__(
            *args, robot_uids="unitree_g1_simplified_upper_body_right_arm", **kwargs
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        super()._initialize_episode(env_idx, options)
        # initialize the robot
        self.agent.robot.set_qpos(self.agent.keyframes["standing"].qpos)
        self.agent.robot.set_pose(self.init_robot_pose)
