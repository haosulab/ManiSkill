import os.path as osp
from collections import OrderedDict
from typing import Any, Dict

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder


# TODO (stao): Complete this task example
# @register_env("FMBAssembly1-v1", max_episode_steps=200)
class FMBAssembly1Env(BaseEnv):
    """
    Task Description
    ----------------
    This task is a simulation version of one of the Multi-Object Multi-Stage Manipulation Tasks (Assembly1) from [Functional Manipulation Benchmark (Luo et. al)](https://functional-manipulation-benchmark.github.io/index.html).
    The goal here is to assemble parts together with the help of a reorientation fixture onto a red board.

    Randomizations
    --------------
    - TODO

    Success Conditions
    ------------------
    - All objects are in the target locations. See Assembly 1 in https://functional-manipulation-benchmark.github.io/files/index.html

    Visualization: TODO: ADD LINK HERE
    """

    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.0, 0.8, 0.8], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 1024, 1024, 1, 0.01, 100)

    def _load_scene(self):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        builder = self._scene.create_actor_builder()

        rot_correction = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0))
        builder.add_nonconvex_collision_from_file(
            osp.join(osp.dirname(__file__), "assets/board_1.glb"), rot_correction
        )
        builder.add_visual_from_file(
            osp.join(osp.dirname(__file__), "assets/board_1.glb"), rot_correction
        )
        self.board = builder.build_kinematic("board")
        builder = self._scene.create_actor_builder()
        builder.add_convex_collision_from_file(
            osp.join(osp.dirname(__file__), "assets/yellow_peg.glb"), rot_correction
        )
        builder.add_visual_from_file(
            osp.join(osp.dirname(__file__), "assets/yellow_peg.glb"), rot_correction
        )
        self.peg = builder.build("yellow_peg")

        builder = self._scene.create_actor_builder()
        builder.add_multiple_convex_collisions_from_file(
            osp.join(osp.dirname(__file__), "assets/purple_u.ply")
        )
        builder.add_visual_from_file(
            osp.join(osp.dirname(__file__), "assets/purple_u.glb"), rot_correction
        )
        self.purple_u = builder.build("purple_u")

        builder = self._scene.create_actor_builder()
        builder.add_multiple_convex_collisions_from_file(
            osp.join(osp.dirname(__file__), "assets/blue_u.ply")
        )
        builder.add_visual_from_file(
            osp.join(osp.dirname(__file__), "assets/blue_u.glb"), rot_correction
        )
        self.blue_u = builder.build("blue_u")

        builder = self._scene.create_actor_builder()
        builder.add_multiple_convex_collisions_from_file(
            osp.join(osp.dirname(__file__), "assets/green_bridge.ply")
        )
        builder.add_visual_from_file(
            osp.join(osp.dirname(__file__), "assets/green_bridge.glb"), rot_correction
        )
        self.bridge = builder.build("green_bridge")

        rot_correction = sapien.Pose(q=euler2quat(np.pi / 2, 0, np.pi / 2))
        builder = self._scene.create_actor_builder()
        builder.add_nonconvex_collision_from_file(
            osp.join(osp.dirname(__file__), "assets/reorienting_fixture.glb"),
            rot_correction,
        )
        builder.add_visual_from_file(
            osp.join(osp.dirname(__file__), "assets/reorienting_fixture.glb"),
            rot_correction,
        )
        self.reorienting_fixture = builder.build_kinematic("reorienting_fixture")

        builder = self._scene.create_actor_builder()
        # builder.add_box_visual(
        #     half_size=[0.02, 0.035, 0.4],
        #     material=sapien.render.RenderMaterial(base_color=[0, 1, 1, 0.3]),
        # )
        self.bridge_grasp = builder.build_kinematic(name="bridge_grasp")

    def _initialize_episode(self, env_idx: torch.Tensor):
        self.table_scene.initialize(env_idx)
        offset_pose = sapien.Pose(p=[0.02, -0.115, 0], q=euler2quat(0, 0, np.pi / 2))
        self.board.set_pose(
            sapien.Pose(p=np.array([0.115, 0.115, 0.034444])) * offset_pose
        )
        self.peg.set_pose(sapien.Pose(p=np.array([0.115, 0.115, 0.0585])) * offset_pose)
        self.purple_u.set_pose(
            sapien.Pose(p=np.array([0.115, 0.047, 0.06375])) * offset_pose
        )
        self.blue_u.set_pose(
            sapien.Pose(p=np.array([0.115, 0.183, 0.06375])) * offset_pose
        )
        self.bridge.set_pose(
            sapien.Pose(p=np.array([0.115, 0.115, 0.048667])) * offset_pose
        )
        self.reorienting_fixture.set_pose(sapien.Pose(p=np.array([0.05, 0.25, 0.0285])))

        # place_order = [self.purple_u, self.blue_u, self.purple_u, self.peg, self.bridge]
        # for i, obj in enumerate(place_order):
        #     obj.set_pose(obj.pose * sapien.Pose([0, 0, 0]))

        self.bridge.set_pose(
            sapien.Pose(
                p=[-0.12, 0.23, 0.048667 / 2], q=euler2quat(0, -np.pi / 2, np.pi / 2)
            )
        )
        self.bridge_grasp_offset = sapien.Pose(p=[0, 0, 0.03])
        self.bridge_grasp.set_pose(self.bridge.pose * self.bridge_grasp_offset)

    def evaluate(self):
        return {"success": torch.zeros(self.num_envs, device=self.device, dtype=bool)}

    def _get_obs_extra(self, info: Dict):
        return OrderedDict()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
