from collections import OrderedDict
from typing import Dict, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots.panda import PandaRealSensed435
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table.table_scene_builder import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.structs.types import SimConfig


def _build_box_with_hole(
    scene: ManiSkillScene, inner_radius, outer_radius, depth, center=(0, 0)
):
    builder = scene.create_actor_builder()
    thickness = (outer_radius - inner_radius) * 0.5
    # x-axis is hole direction
    half_center = [x * 0.5 for x in center]
    half_sizes = [
        [depth, thickness - half_center[0], outer_radius],
        [depth, thickness + half_center[0], outer_radius],
        [depth, outer_radius, thickness - half_center[1]],
        [depth, outer_radius, thickness + half_center[1]],
    ]
    offset = thickness + inner_radius
    poses = [
        sapien.Pose([0, offset + half_center[0], 0]),
        sapien.Pose([0, -offset + half_center[0], 0]),
        sapien.Pose([0, 0, offset + half_center[1]]),
        sapien.Pose([0, 0, -offset + half_center[1]]),
    ]

    mat = sapien.render.RenderMaterial(
        base_color=sapien_utils.hex2rgba("#FFD289"), roughness=0.5, specular=0.5
    )

    for half_size, pose in zip(half_sizes, poses):
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=mat)
    return builder


@register_env("PegInsertionSide-v1", max_episode_steps=200)
class PegInsertionSideEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["sparse", "none"]
    SUPPORTED_ROBOTS = ["panda_realsensed435"]
    agent: Union[PandaRealSensed435]
    _clearance = 0.003

    def __init__(self, *args, robot_uids="panda_realsensed435", num_envs=1, **kwargs):
        reconfiguration_freq = 0
        if num_envs == 1:
            reconfiguration_freq = 1
        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

    @property
    def _default_sim_cfg(self):
        return SimConfig()

    @property
    def _sensor_configs(self):
        pose = sapien_utils.look_at([0, -0.3, 0.2], [0, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.5, -0.5, 0.8], [0.05, -0.1, 0.4])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self):
        with torch.device(self.device):
            self.table_scene = TableSceneBuilder(self)
            self.table_scene.build()

            lengths = self._episode_rng.uniform(0.085, 0.125, size=(self.num_envs,))
            radii = self._episode_rng.uniform(0.015, 0.025, size=(self.num_envs,))
            centers = (
                0.5
                * (lengths - radii)[:, None]
                * self._episode_rng.uniform(-1, 1, size=(self.num_envs, 2))
            )

            # save some useful values for use later
            self.peg_half_sizes = sapien_utils.to_tensor(
                np.vstack([lengths, radii, radii])
            ).T
            peg_head_offsets = torch.zeros((self.num_envs, 3))
            peg_head_offsets[:, 0] = self.peg_half_sizes[:, 0]
            self.peg_head_offsets = Pose.create_from_pq(p=peg_head_offsets)

            box_hole_offsets = torch.zeros((self.num_envs, 3))
            box_hole_offsets[:, 1:] = sapien_utils.to_tensor(centers)
            self.box_hole_offsets = Pose.create_from_pq(p=box_hole_offsets)
            self.box_hole_radii = sapien_utils.to_tensor(radii + self._clearance)

            # in each parallel env we build a different box with a hole and peg (the task is meant to be quite difficult)
            pegs = []
            boxes = []

            for i in range(self.num_envs):
                scene_idxs = [i]
                length = lengths[i]
                radius = radii[i]
                builder = self._scene.create_actor_builder()
                builder.add_box_collision(half_size=[length, radius, radius])
                # peg head
                mat = sapien.render.RenderMaterial(
                    base_color=sapien_utils.hex2rgba("#EC7357"),
                    roughness=0.5,
                    specular=0.5,
                )
                builder.add_box_visual(
                    sapien.Pose([length / 2, 0, 0]),
                    half_size=[length / 2, radius, radius],
                    material=mat,
                )
                # peg tail
                mat = sapien.render.RenderMaterial(
                    base_color=sapien_utils.hex2rgba("#EDF6F9"),
                    roughness=0.5,
                    specular=0.5,
                )
                builder.add_box_visual(
                    sapien.Pose([-length / 2, 0, 0]),
                    half_size=[length / 2, radius, radius],
                    material=mat,
                )
                builder.set_scene_idxs(scene_idxs)
                peg = builder.build(f"peg_{i}")

                # box with hole

                inner_radius, outer_radius, depth = (
                    radius + self._clearance,
                    length,
                    length,
                )
                builder = _build_box_with_hole(
                    self._scene, inner_radius, outer_radius, depth, center=centers[i]
                )
                builder.set_scene_idxs(scene_idxs)
                box = builder.build_kinematic(f"box_with_hole_{i}")

                pegs.append(peg)
                boxes.append(box)
            self.peg = Actor.merge(pegs, "peg")
            self.box = Actor.merge(boxes, "box_with_hole")

    def _initialize_episode(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # initialize the box and peg
            xy = torch.rand((b, 2)) * torch.tensor([0.175, 0.3]) + torch.tensor(
                [0, -0.15]
            )
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = self.peg_half_sizes[:, 2]
            quat = randomization.random_quaternions(
                b,
                self.device,
                lock_x=True,
                lock_y=True,
                bounds=(np.pi / 2 - np.pi / 3, np.pi / 2 - np.pi / 3),
            )
            self.peg.set_pose(Pose.create_from_pq(pos, quat))

            xy = torch.rand((b, 2)) * torch.tensor([0.1, 0.2]) + torch.tensor(
                [-0.05, 0.2]
            )
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = self.peg_half_sizes[:, 0]
            quat = randomization.random_quaternions(
                b,
                self.device,
                lock_x=True,
                lock_y=True,
                bounds=(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8),
            )
            self.box.set_pose(Pose.create_from_pq(pos, quat))

            # Initialize the robot
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    -np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            qpos = self._episode_rng.normal(0, 0.02, (b, len(qpos))) + qpos
            qpos[:, -2:] = 0.04
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

    # save some commonly used attributes
    @property
    def peg_head_pos(self):
        return self.peg.pose.p + self.peg_head_offsets.p

    @property
    def peg_head_pose(self):
        return self.peg.pose * self.peg_head_offsets

    @property
    def box_hole_pose(self):
        return self.box.pose * self.box_hole_offsets

    @property
    def goal_pose(self):
        # NOTE (stao): this is fixed after each _initialize_episode call. You can cache this value
        # and simply store it after _initialize_episode or set_state_dict calls.
        return self.box.pose * self.box_hole_offsets * self.peg_head_offsets.inv()

    def has_peg_inserted(self):
        # Only head position is used in fact
        peg_head_pos_at_hole = (self.box_hole_pose.inv() * self.peg_head_pose).p
        # x-axis is hole direction
        x_flag = -0.015 <= peg_head_pos_at_hole[:, 0]
        y_flag = (-self.box_hole_radii <= peg_head_pos_at_hole[:, 1]) & (
            peg_head_pos_at_hole[:, 1] <= self.box_hole_radii
        )
        z_flag = (-self.box_hole_radii <= peg_head_pos_at_hole[:, 2]) & (
            peg_head_pos_at_hole[:, 2] <= self.box_hole_radii
        )
        return (
            x_flag & y_flag & z_flag,
            peg_head_pos_at_hole,
        )

    def evaluate(self):
        success, peg_head_pos_at_hole = self.has_peg_inserted()
        return dict(success=success, peg_head_pos_at_hole=peg_head_pos_at_hole)

    def _get_obs_extra(self, info: Dict):
        obs = OrderedDict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                peg_pose=self.peg.pose.raw_pose,
                peg_half_size=self.peg_half_sizes,
                box_hole_pose=self.box_hole_pose.raw_pose,
                box_hole_radius=self.box_hole_radii,
            )
        return obs

    # TODO (stao): Write batched version of dense reward function
    # Code here: https://github.com/haosulab/ManiSkill2/blob/4067abce40dc8b7d7c926eabc9c48e802489014c/mani_skill2/envs/ms2/assembly/peg_insertion_side.py#L154
