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
    A base environment for simulating manipulation tasks in more complex scenes. Creating this base environment is only useful for explorations/visualization, there are no success/failure
    metrics or rewards.

    Args:
        robot_uids: Which robot to place into the scene. Default is "panda"

        fixed_scene: whether to sample a single scene and never reconfigure the scene during episode resets
        Default to True as reconfiguration/reloading scenes is expensive. When true, call env.reset(seed=seed, options=dict(reconfigure=True))

        scene_builder_cls: Scene builder class to build a scene with. Default is the ArchitecTHORSceneBuilder which builds a scene from AI2THOR.
            Any of the AI2THOR SceneBuilders are supported in this environment

        convex_decomposition: Choice of convex decomposition algorithm to generate collision meshes for objects. Default is `coacd` which uses https://github.com/SarahWeiii/CoACD
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self,
        *args,
        robot_uids="fetch",
        robot_init_qpos_noise=0.02,
        fixed_scene=True,
        scene_builder_cls: Union[str, SceneBuilder] = "ReplicaCAD",
        convex_decomposition="coacd",
        scene_idxs=None,
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.fixed_scene = fixed_scene
        self.sampled_scene_idx: int = None
        if isinstance(scene_builder_cls, str):
            scene_builder_cls = REGISTERED_SCENE_BUILDERS[
                scene_builder_cls
            ].scene_builder_cls
        self.scene_builder: SceneBuilder = scene_builder_cls(self)
        if isinstance(scene_idxs, int):
            self.scene_idxs = [scene_idxs]
        elif isinstance(scene_idxs, list):
            self.scene_idxs = scene_idxs
        else:
            self.scene_idxs = np.arange(0, len(self.scene_builder.scene_configs))
        self.convex_decomposition = convex_decomposition
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_cfg(self):
        return SimConfig(
            spacing=50,
            gpu_memory_cfg=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**19,
                max_rigid_contact_count=2**21,
            ),
        )

    def reset(self, seed=None, options=None):
        self._set_episode_rng(seed)
        if options is None:
            options = dict(reconfigure=False)
        if not self.fixed_scene:
            options["reconfigure"] = True
        if "reconfigure" in options and options["reconfigure"]:
            self.sampled_scene_idx = self.scene_idxs[
                self._episode_rng.randint(0, len(self.scene_idxs))
            ]
            self.sampled_scene_idx = int(self.sampled_scene_idx)
        return super().reset(seed, options)

    def _load_lighting(self):
        if self.scene_builder.builds_lighting:
            return
        return super()._load_lighting()

    def _load_scene(self):
        self.scene_builder.build(
            self._scene,
            scene_idx=self.sampled_scene_idx,
            convex_decomposition=self.convex_decomposition,
        )

    def _initialize_episode(self, env_idx: torch.Tensor):
        with torch.device(self.device):
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
    def _sensor_configs(self):
        if self.robot_uids == "fetch":
            return ()

        pose = sapien_utils.look_at([0.3, 0, 0.6], [-0.1, 0, 0.1])
        return CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)

    @property
    def _human_render_camera_configs(self):
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
