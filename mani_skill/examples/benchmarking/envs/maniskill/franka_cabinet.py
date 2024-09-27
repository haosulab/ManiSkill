from typing import Any, Dict
import numpy as np
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.mobile_manipulation.open_cabinet_drawer import OpenCabinetDrawerEnv
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.building.ground import build_ground

from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SceneConfig, SimConfig
CABINET_COLLISION_BIT=29
@register_env("FrankaCabinetBenchmarkEnv-v1")
class FrankaCabinetBenchmarkEnv(BaseEnv):
    def __init__(self, *args, robot_uids="panda", camera_width=128, camera_height=128, num_cameras=1, **kwargs):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.num_cameras = num_cameras
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=120,
            control_freq=60,
            scene_config=SceneConfig(
                # bounce_threshold=0.01,
                solver_position_iterations=12, solver_velocity_iterations=1
            ),
        )
    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0, -4, 1], target=[0, 0, 1])
        sensor_configs = []
        if self.num_cameras is not None:
            for i in range(self.num_cameras):
                sensor_configs.append(CameraConfig(uid=f"base_camera_{i}",
                                                pose=pose,
                                                width=self.camera_width,
                                                height=self.camera_height,
                                                fov=np.pi / 2))
        return sensor_configs

    def evaluate(self):
        return {}
    def _get_obs_extra(self, info: Dict):
        return {}

    def _load_scene(self, options: dict):
        cabinet_builder = articulations.get_articulation_builder(
            self.scene, f"partnet-mobility:{1052}"
        )
        cabinet_builder.disable_self_collisions = True
        self.cabinet = cabinet_builder.build(name="cabinet", fix_root_link=True)

        # disable cabinet self collisions
        for link in self.cabinet.get_links():
            link.set_collision_group_bit(
                group=2, bit_idx=CABINET_COLLISION_BIT, bit=1
            )
        # disable robot self collisions
        for link in self.agent.robot.get_links():
            link.set_collision_group_bit(
                group=2, bit_idx=28, bit=1
            )
        self.ground = build_ground(self.scene)
        self.ground.set_collision_group_bit(
            group=2, bit_idx=CABINET_COLLISION_BIT, bit=1
        )
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.45
            self.cabinet.set_pose(Pose.create_from_pq(p=xyz))
            xyz = torch.zeros((b, 3))
            xyz[:, 0] = -1.2
            xyz[:, 2] = 0
            self.agent.robot.set_pose(Pose.create_from_pq(p=xyz))
            self.agent.robot.set_qpos(self.agent.keyframes["rest"].qpos)

    def compute_dense_reward(self, obs: torch.Any, action: torch.Tensor, info: torch.Dict):
        return torch.zeros((self.num_envs, ), device=self.device)
    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return self.compute_dense_reward(obs, action, info)
