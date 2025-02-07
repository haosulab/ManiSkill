from typing import Any, Dict, List, Union

import numpy as np
import torch

from mani_skill import logger
from mani_skill.agents.robots import DClaw
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.articulations import build_robel_valve
from mani_skill.utils.geometry.rotation_conversions import axis_angle_to_quaternion
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.pose import Pose, vectorize_pose
from mani_skill.utils.structs.types import Array


class RotateValveEnv(BaseEnv):
    agent: Union[DClaw]
    _clearance = 0.003

    def __init__(
        self,
        *args,
        robot_init_qpos_noise=0.02,
        valve_init_pos_noise=0.02,
        difficulty_level: int = -1,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.valve_init_pos_noise = valve_init_pos_noise

        if (
            not isinstance(difficulty_level, int)
            or difficulty_level >= 5
            or difficulty_level < 0
        ):
            raise ValueError(
                f"Difficulty level must be a int within 0-4, but get {difficulty_level}"
            )
        self.difficulty_level = difficulty_level

        # Task information
        # For the simplest level 0, only quarter round will make it a success
        # For the hardest level 4, rotate one rounds will make it a success
        # For other intermediate level 1-3, the success threshold should be half round
        if self.difficulty_level == 0:
            self.success_threshold = torch.pi / 2
        elif self.difficulty_level == 4:
            self.success_threshold = torch.pi * 2
        else:
            self.success_threshold = torch.pi * 1

        self.capsule_offset = 0.01

        super().__init__(*args, robot_uids="dclaw", **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.3], target=[-0.1, 0, 0.05])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.2, 0.4, 0.4], [0.0, 0.0, 0.1])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self._load_articulations()

    def _load_articulations(self):
        # Robel valve
        if self.difficulty_level == 0:
            # Only tri-valve
            valve_angles_list = [(0, np.pi / 3 * 2, np.pi / 3 * 4)] * self.num_envs
        elif self.difficulty_level == 1:
            base_angles = [
                np.arange(0, np.pi * 2, np.pi * 2 / 3),
                np.arange(0, np.pi * 2, np.pi / 2),
                np.arange(0, np.pi * 2, np.pi * 2 / 5),
            ]
            valve_angles_list = (
                base_angles * int(self.num_envs // 3)
                + base_angles[: int(self.num_envs % 3)]
            )
        elif self.difficulty_level == 2:
            num_valve_head = self._batched_episode_rng.randint(3, 6)
            valve_angles_list = [
                sample_valve_angles(num_head, self._batched_episode_rng[i])
                for i, num_head in enumerate(num_valve_head)
            ]
        elif self.difficulty_level >= 3:
            num_valve_head = self._batched_episode_rng.randint(3, 6)
            valve_angles_list = [
                sample_valve_angles(num_head, self._batched_episode_rng[i])
                for i, num_head in enumerate(num_valve_head)
            ]
        else:
            raise ValueError(
                f"Difficulty level must be a int within 0-4, but get {self.difficulty_level}"
            )

        valves: List[Articulation] = []
        capsule_lens = []
        valve_links = []
        for i, valve_angles in enumerate(valve_angles_list):
            scene_idxs = [i]
            if self.difficulty_level < 3:
                valve, capsule_len = build_robel_valve(
                    self.scene,
                    valve_angles=valve_angles,
                    scene_idxs=scene_idxs,
                    name=f"valve_station_{i}",
                )
            else:
                scales = self._batched_episode_rng[i].randn(2) * 0.1 + 1
                valve, capsule_len = build_robel_valve(
                    self.scene,
                    valve_angles=valve_angles,
                    scene_idxs=scene_idxs,
                    name=f"valve_station_{i}",
                    radius_scale=scales[0],
                    capsule_radius_scale=scales[1],
                )
            valves.append(valve)
            valve_links.append(valve.links_map["valve"])
            capsule_lens.append(capsule_len)
        self.valve = Articulation.merge(valves, "valve_station")
        self.capsule_lens = torch.from_numpy(np.array(capsule_lens)).to(self.device)
        self.valve_link = Link.merge(valve_links, name="valve")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._initialize_actors(env_idx)
        self._initialize_agent(env_idx)

    def _initialize_actors(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Initialize task related information
            if self.difficulty_level <= 3:
                self.rotate_direction = torch.ones(b)
            else:
                self.rotate_direction = 1 - torch.randint(0, 2, (b,)) * 2

            # Initialize the valve
            xyz = torch.zeros((b, 3))
            xyz[:, :2].uniform_(-0.02, 0.02)
            axis_angle = torch.zeros((b, 3))
            axis_angle[:, 2].uniform_(torch.pi / 6, torch.pi * 5 / 6)
            pose = Pose.create_from_pq(xyz, axis_angle_to_quaternion(axis_angle))
            self.valve.set_pose(pose)

            qpos = torch.rand((b, 1)) * torch.pi * 2 - torch.pi
            self.valve.set_qpos(qpos)
            self.rest_qpos = qpos

    def _initialize_agent(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)
            dof = self.agent.robot.dof
            if isinstance(dof, torch.Tensor):
                dof = dof[0]

            init_qpos = torch.zeros((b, dof))
            # set root joint qpos to avoid robot-object collision after reset
            init_qpos[:, self.agent.root_joint_indices] = torch.tensor(
                [0.7, -0.7, -0.7]
            )
            init_qpos += torch.randn((b, dof)) * self.robot_init_qpos_noise
            self.agent.reset(init_qpos)
            self.agent.robot.set_pose(
                Pose.create_from_pq(
                    torch.tensor([0.0, 0, 0.28]), torch.tensor([0, 0, -1, 0])
                )
            )

    def _get_obs_extra(self, info: Dict):
        with torch.device(self.device):
            valve_qpos = self.valve.qpos
            valve_qvel = self.valve.qvel
            obs = dict(
                rotate_dir=self.rotate_direction.to(torch.float32),
                valve_qpos=valve_qpos,
                valve_qvel=valve_qvel,
                valve_x=torch.cos(valve_qpos[:, 0]),
                valve_y=torch.sin(valve_qpos[:, 0]),
            )
            if self.obs_mode_struct.use_state:
                obs.update(
                    valve_pose=vectorize_pose(self.valve.pose),
                )
            return obs

    def evaluate(self, **kwargs) -> dict:
        valve_rotation = (self.valve.qpos - self.rest_qpos)[:, 0]
        success = valve_rotation * self.rotate_direction > self.success_threshold
        return dict(success=success, valve_rotation=valve_rotation)

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        rotation = info["valve_rotation"]
        qvel = self.valve.qvel

        # Distance between fingertips and the circle grouned by valve tips
        tip_poses = self.agent.tip_poses  # (b, 3, 7)
        tip_pos = tip_poses[:, :, :2]  # (b, 3, 2)
        valve_pos = self.valve_link.pose.p[:, :2]  # (b, 2)
        valve_tip_dist = torch.linalg.norm(tip_pos - valve_pos[:, None, :], dim=-1)
        desired_valve_tip_dist = self.capsule_lens[:, None] - self.capsule_offset
        error = torch.norm(valve_tip_dist - desired_valve_tip_dist, dim=-1)
        reward = 1 - torch.tanh(error * 10)

        directed_velocity = qvel[:, 0] * self.rotate_direction
        reward += torch.tanh(5 * directed_velocity) * 4

        motion_reward = torch.clip(rotation / torch.pi / 2, -1, 1)
        reward += motion_reward

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6.0


def sample_valve_angles(
    num_head: int,
    random_state: np.random.RandomState,
    min_angle_diff=np.pi / 6,
    num_max_attempts=500,
):
    for i in range(num_max_attempts):
        angles = random_state.uniform(0, np.pi * 2, (num_head,))
        angles = np.sort(angles)

        # Append a 360 degree at the end of the list to check the last angle with the first one (0-degree)
        diff = np.append(angles[1:], np.pi * 2) - angles
        if np.min(diff) >= min_angle_diff:
            return angles

    logger.warn(
        f"sample_valve_angles reach max attempts {num_max_attempts}. Will use the default valve angles."
    )
    return np.arange(0, np.pi * 2, np.pi * 2 / num_head)


@register_env("RotateValveLevel0-v1", max_episode_steps=80)
class RotateValveEnvLevel0(RotateValveEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            robot_init_qpos_noise=0.02,
            valve_init_pos_noise=0.02,
            difficulty_level=0,
            **kwargs,
        )


@register_env("RotateValveLevel1-v1", max_episode_steps=150)
class RotateValveEnvLevel1(RotateValveEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            robot_init_qpos_noise=0.02,
            valve_init_pos_noise=0.02,
            difficulty_level=1,
            **kwargs,
        )


@register_env("RotateValveLevel2-v1", max_episode_steps=150)
class RotateValveEnvLevel2(RotateValveEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            robot_init_qpos_noise=0.02,
            valve_init_pos_noise=0.02,
            difficulty_level=2,
            **kwargs,
        )


@register_env("RotateValveLevel3-v1", max_episode_steps=150)
class RotateValveEnvLevel3(RotateValveEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            robot_init_qpos_noise=0.02,
            valve_init_pos_noise=0.02,
            difficulty_level=3,
            **kwargs,
        )


@register_env("RotateValveLevel4-v1", max_episode_steps=300)
class RotateValveEnvLevel4(RotateValveEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            robot_init_qpos_noise=0.02,
            valve_init_pos_noise=0.02,
            difficulty_level=4,
            **kwargs,
        )
