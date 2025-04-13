from typing import Any, Dict, Union

import numpy as np
import sapien as sapien
import sapien.physx as physx
import torch
from sapien import Pose

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder import SceneBuilder
from mani_skill.utils.scene_builder.registration import REGISTERED_SCENE_BUILDERS
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@register_env("SceneManipulation-v1", max_episode_steps=200)
class SceneManipulationEnv(BaseEnv):
    """
    A base environment for simulating manipulation tasks in more complex scenes. Creating this base environment is only useful
    for explorations/visualization, there are no success/failure metrics or rewards.

    Args:
        robot_uids: Which robot to place into the scene. Default is "fetch"

        fixed_scene:
            When True, will never reconfigure the environment during resets unless you run env.reset(seed=seed, options=dict(reconfigure=True))
            and explicitly reconfigure. If False, will reconfigure every reset.

        scene_builder_cls:
            Scene builder class to build a scene with. Default is ReplicaCAD. Furthermore, any of the AI2THOR SceneBuilders are supported in
            this environment.

        build_config_idxs (optional): which build configs (static builds) to sample. Your scene_builder_cls may or may not require these.
        init_config_idxs (optional): which init configs (additional init options) to sample. Your scene_builder_cls may or may not require these.
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self,
        *args,
        robot_uids="fetch",
        scene_builder_cls: Union[str, SceneBuilder] = "ReplicaCAD",
        build_config_idxs=None,
        init_config_idxs=None,
        num_envs=1,
        reconfiguration_freq=None,
        **kwargs
    ):
        if isinstance(scene_builder_cls, str):
            scene_builder_cls = REGISTERED_SCENE_BUILDERS[
                scene_builder_cls
            ].scene_builder_cls
        self.scene_builder: SceneBuilder = scene_builder_cls(self)
        self.build_config_idxs = build_config_idxs
        self.init_config_idxs = init_config_idxs
        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            spacing=50,
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**21,
                max_rigid_contact_count=2**23,
            ),
        )

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict(reconfigure=False)
        self._set_episode_rng(seed, options.get("env_idx", torch.arange(self.num_envs)))
        if "reconfigure" in options and options["reconfigure"]:
            self.build_config_idxs = options.get(
                "build_config_idxs", self.build_config_idxs
            )
            self.init_config_idxs = options.get("init_config_idxs", None)
        else:
            assert (
                "build_config_idxs" not in options
            ), "options dict cannot contain build_config_idxs without reconfigure=True"
            self.init_config_idxs = options.get(
                "init_config_idxs", self.init_config_idxs
            )
        if isinstance(self.build_config_idxs, int):
            self.build_config_idxs = [self.build_config_idxs]
        if isinstance(self.init_config_idxs, int):
            self.init_config_idxs = [self.init_config_idxs]
        return super().reset(seed, options)

    def _load_lighting(self, options: dict):
        if self.scene_builder.builds_lighting:
            return
        return super()._load_lighting(options)

    def _load_agent(self, options: dict):
        super()._load_agent(
            options,
            self.scene_builder.robot_initial_pose,
        )

    def _load_scene(self, options: dict):
        if self.scene_builder.build_configs is not None:
            self.scene_builder.build(
                self.build_config_idxs
                if self.build_config_idxs is not None
                else self.scene_builder.sample_build_config_idxs()
            )
        else:
            self.scene_builder.build()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            if self.scene_builder.init_configs is not None:
                self.scene_builder.initialize(
                    env_idx,
                    (
                        self.init_config_idxs
                        if self.init_config_idxs is not None
                        else self.scene_builder.sample_init_config_idxs()
                    ),
                )
            else:
                self.scene_builder.initialize(env_idx)

    def evaluate(self) -> dict:
        return dict()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 0

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 1

    @property
    def _default_sensor_configs(self):
        if self.robot_uids == "fetch":
            return []

        pose = sapien_utils.look_at([0.3, 0, 0.6], [-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        if self.robot_uids == "fetch":
            room_camera_pose = sapien_utils.look_at([2.5, -2.5, 3], [0.0, 0.0, 0])
            room_camera_config = CameraConfig(
                "render_camera",
                room_camera_pose,
                512,
                512,
                1,
                0.01,
                100,
            )
            robot_camera_pose = sapien_utils.look_at([2, 0, 1], [0, 0, -1])
            robot_camera_config = CameraConfig(
                "robot_render_camera",
                robot_camera_pose,
                512,
                512,
                1.5,
                0.01,
                100,
                mount=self.agent.torso_lift_link,
            )
            return [room_camera_config, robot_camera_config]

        if self.robot_uids == "panda":
            pose = sapien_utils.look_at([0.4, 0.4, 0.8], [0.0, 0.0, 0.4])
        else:
            pose = sapien_utils.look_at([0, 10, -3], [0, 0, 0])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)
