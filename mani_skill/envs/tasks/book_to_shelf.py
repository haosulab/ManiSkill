from pathlib import Path
from collections import OrderedDict
from typing import Dict, Union

import numpy as np
import torch
import torch.random
from transforms3d.euler import euler2quat
import sapien
import sapien.physx as physx
import sapien.render
import transforms3d

from mani_skill.agents.robots import Fetch, Panda, Xmate3Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.scene_builder.table.table_scene_builder import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array


@register_env("BookToShelf-v0", max_episode_steps=50)
class BookToShelfEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = [
        "sparse",
        "none",
    ]  # TODO add a denser reward for this later
    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]
    agent: Union[Panda, Xmate3Robotiq, Fetch]

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _sensor_configs(self):
        pose = look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _human_render_camera_configs(self):
        pose = look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.book = actors.build_actor_ai2_helper(
            model_id="Book_14",
            scene=self._scene,
            name="book",
            kinematic=False,
            set_object_on_ground=False,
        )

        self.shelf = actors.build_actor_ai2_helper(
            model_id="FloorPlan303_physics-Shelf",
            scene=self._scene,
            name="shelf",
            kinematic=True,
            set_object_on_ground=False,
        )

    def _initialize_episode(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[..., 2] = 0.075
            q = euler2quat(0, np.pi/2, np.pi/2)

            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.book.set_pose(obj_pose)
            xyz[..., 1] -= 0.3
            xyz[..., 2] += 0.15
            q = euler2quat(0, 0, np.pi)
            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.shelf.set_pose(obj_pose)

    def evaluate(self):
        is_book_placed = (
            torch.linalg.norm(
                self.book.pose.p - self.shelf.pose.p, axis=1
            )
            < 0.05
        )

        return {
            "success": is_book_placed,
        }

    def _get_obs_extra(self, info: Dict):
        obs = OrderedDict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                obj_pose=self.book.pose.raw_pose,
            )
        return obs