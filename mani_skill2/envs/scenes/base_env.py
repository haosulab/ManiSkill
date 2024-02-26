from typing import Union

import numpy as np
import sapien as sapien
import sapien.physx as physx
import torch
from sapien import Pose

from mani_skill2.agents.robots import Fetch, Panda
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.sapien_utils import look_at
from mani_skill2.utils.scene_builder import SceneBuilder
from mani_skill2.utils.scene_builder.ai2thor import (
    ArchitecTHORSceneBuilder,
    ProcTHORSceneBuilder,
    RoboTHORSceneBuilder,
    iTHORSceneBuilder,
)
from mani_skill2.utils.structs.pose import vectorize_pose


class SceneManipulationEnv(BaseEnv):
    agent: Union[Panda, Fetch]
    """
    Args:
        robot_uids: Which robot to place into the scene. Default is "panda"

        fixed_scene: whether to sample a single scene and never reconfigure the scene during episode resets
        Default to True as reconfiguration/reloading scenes is expensive. When true, call env.reset(seed=seed, options=dict(reconfigure=True))

        scene_builder_cls: Scene builder class to build a scene with. Default is the ArchitecTHORSceneBuilder which builds a scene from AI2THOR.
            Any of the AI2THOR SceneBuilders are supported in this environment

        convex_decomposition: Choice of convex decomposition algorithm to generate collision meshes for objects. Default is `coacd` which uses https://github.com/SarahWeiii/CoACD
    """

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        fixed_scene=True,
        scene_builder_cls: SceneBuilder = ArchitecTHORSceneBuilder,
        convex_decomposition="coacd",
        scene_idxs=None,
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.fixed_scene = fixed_scene
        self.sampled_scene_idx: int = None
        self.scene_builder: SceneBuilder = scene_builder_cls(
            self, robot_init_qpos_noise=robot_init_qpos_noise
        )
        if isinstance(scene_idxs, int):
            self.scene_idxs = [scene_idxs]
        elif isinstance(scene_idxs, list):
            self.scene_idxs = scene_idxs
        else:
            self.scene_idxs = np.arange(0, len(self.scene_builder.scene_configs))
        self.convex_decomposition = convex_decomposition
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

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
        return super().reset(seed, options)

    def _load_actors(self):
        self.scene_builder.build(
            self._scene,
            scene_idx=self.sampled_scene_idx,
            convex_decomposition=self.convex_decomposition,
        )

    def _initialize_actors(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            self.scene_builder.initialize(env_idx)

    def _register_sensors(self):
        if self.robot_uids == "fetch":
            return ()

        pose = look_at([0.3, 0, 0.6], [-0.1, 0, 0.1])
        return CameraConfig(
            "base_camera", pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10
        )

    def _register_human_render_cameras(self):
        if self.robot_uids == "fetch":
            room_camera_pose = look_at([-6, 2, 2], [-2.5, 2, 0])
            room_camera_config = CameraConfig(
                "render_camera",
                room_camera_pose.p,
                room_camera_pose.q,
                512,
                512,
                1,
                0.01,
                10,
            )
            robot_camera_pose = look_at([2, 0, 1], [0, 0, -1])
            robot_camera_config = CameraConfig(
                "robot_render_camera",
                robot_camera_pose.p,
                robot_camera_pose.q,
                512,
                512,
                1.5,
                0.01,
                10,
                link=self.agent.torso_lift_link,
            )
            return room_camera_config, robot_camera_config

        if self.robot_uids == "panda":
            pose = look_at([0.4, 0.4, 0.8], [0.0, 0.0, 0.4])
        else:
            pose = look_at([0, 10, -3], [0, 0, 0])
        return CameraConfig("render_camera", pose.p, pose.q, 512, 512, 1, 0.01, 10)

    def _setup_viewer(self):
        super()._setup_viewer()
        self._viewer.set_camera_xyz(0.8, 0, 1.0)
        self._viewer.set_camera_rpy(0, -0.5, 3.14)

    def _get_obs_agent(self):
        obs = self.agent.get_proprioception()
        return obs
