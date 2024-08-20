from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import UnitreeG1Simplified, UnitreeH1Simplified
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


class HumanoidStandEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["sparse", "none"]

    def __init__(
        self,
        *args,
        robot_uids="unitree_h1_simplified",
        robot_init_qpos_noise=0.02,
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        return []

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.0, 1.0, 2.5], [0.0, 0.0, 0.75])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        build_ground(self.scene)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        pass

    def evaluate(self):
        is_standing = self.agent.is_standing()
        self.agent.is_fallen()
        return {"is_standing": is_standing, "fail": ~is_standing}

    def _get_obs_extra(self, info: Dict):
        return dict()

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return info["is_standing"]

    # def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
    #     return torch.zeros(self.num_envs, device=self.device)

    # def compute_normalized_dense_reward(
    #     self, obs: Any, action: torch.Tensor, info: Dict
    # ):
    #     max_reward = 1.0
    #     return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward


# Different robot embodiments require different configurations for optimal running and nicer render videos, we define those specifics below
@register_env("UnitreeH1Stand-v1", max_episode_steps=1000)
class UnitreeH1StandEnv(HumanoidStandEnv):
    SUPPORTED_ROBOTS = ["unitree_h1_simplified"]
    agent: Union[UnitreeH1Simplified]

    def __init__(self, *args, robot_uids="unitree_h1_simplified", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**22, max_rigid_patch_count=2**21
            )
        )

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.0, 1.0, 2.5], [0.0, 0.0, 0.75])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            standing_keyframe = self.agent.keyframes["standing"]
            random_qpos = (
                torch.randn(size=(b, self.agent.robot.dof[0]), dtype=torch.float) * 0.05
            )
            random_qpos += common.to_tensor(standing_keyframe.qpos, device=self.device)
            self.agent.robot.set_qpos(random_qpos)
            self.agent.robot.set_pose(sapien.Pose(p=[0, 0, 0.975]))


@register_env("UnitreeG1Stand-v1", max_episode_steps=1000)
class UnitreeG1StandEnv(HumanoidStandEnv):
    SUPPORTED_ROBOTS = ["unitree_g1_simplified_legs"]
    agent: Union[UnitreeG1Simplified]

    def __init__(self, *args, robot_uids="unitree_g1_simplified_legs", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**22, max_rigid_patch_count=2**21
            )
        )

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.0, 1.0, 2.0], [0.0, 0.0, 0.75])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            standing_keyframe = self.agent.keyframes["standing"]
            random_qpos = (
                torch.randn(size=(b, self.agent.robot.dof[0]), dtype=torch.float) * 0.05
            )
            random_qpos += common.to_tensor(standing_keyframe.qpos, device=self.device)
            self.agent.robot.set_qpos(random_qpos)
            self.agent.robot.set_pose(sapien.Pose(p=[0, 0, 0.755]))
