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

from mani_skill.agents.robots import Fetch, Panda, Xmate3Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.scene_builder.table.table_scene_builder import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("PourBowlContents-v0", max_episode_steps=50)
class PourBowlContentsEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = [
        "sparse",
        "none",
    ]  # TODO add a denser reward for this later
    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]
    agent: Union[Panda, Xmate3Robotiq, Fetch]

    peg_half_length = 0.12

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

        self.egg = actors.build_actor_ai2_helper(
            model_id="Egg_1",
            scene=self._scene,
            name="egg",
            kinematic=False,
            set_object_on_ground=False,
        )

        self.bowl1 = actors.build_actor_ai2_helper(
            model_id="Bowl_3",
            scene=self._scene,
            name="bowl1",
            kinematic=False,
            set_object_on_ground=False,
        )

        self.bowl2 = actors.build_actor_ai2_helper(
            model_id="Bowl_11",
            scene=self._scene,
            name="bowl2",
            kinematic=True,
            set_object_on_ground=False,
        )

    def _initialize_episode(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz_b = torch.zeros((b, 3))
            xyz_b[..., :2] = torch.rand((b, 2)) * 0.2 - 0.2
            q = euler2quat(0, 0, 0)
            xyz_b[..., 2] = 0
            xyz_e = xyz_b

            obj_pose = Pose.create_from_pq(p=xyz_b, q=q)
            self.bowl1.set_pose(obj_pose)

            xyz_e[..., 2] += 0.05
            obj_pose = Pose.create_from_pq(p=xyz_e, q=q)
            self.egg.set_pose(obj_pose)

            xyz_b[..., 1] += 0.2
            xyz_b[..., 2] -= 0.05
            obj_pose = Pose.create_from_pq(p=xyz_b, q=q)
            self.bowl2.set_pose(obj_pose)

    def evaluate(self):
        is_egg_placed = (
            torch.linalg.norm(
                self.egg.pose.p - self.bowl2.pose.p, axis=1
            )
            < 0.05
        )

        return {
            "success": is_egg_placed,
        }

    def _get_obs_extra(self, info: Dict):
        obs = OrderedDict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                obj_pose=self.egg.pose.raw_pose,
            )
        return obs