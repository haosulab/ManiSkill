from typing import Any, Dict, Union

import numpy as np
import torch

from mani_skill.agents.robots import Fetch, Panda, Xmate3Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("TranslatePeg-v1", max_episode_steps=50)
class TranslatePegEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]
    agent: Union[Panda, Xmate3Robotiq, Fetch]

    cube_half_size = 0.02
    peg_half_width = 0.025
    peg_half_length = 0.12

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.peg = actors.build_twocolor_peg(
            self.scene,
            length=self.peg_half_length,
            width=self.peg_half_width,
            color_1=np.array([176, 14, 14, 255]) / 255,
            color_2=np.array([12, 42, 160, 255]) / 255,
            name="peg",
            body_type="dynamic",
        )

        self.goal_peg = actors.build_twocolor_peg(
            self.scene,
            length=self.peg_half_length,
            width=self.peg_half_width,
            color_1=[0, 1, 0, 1],
            color_2=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            # initialize the peg
            peg_xyz = torch.zeros((b, 3))
            peg_xyz = torch.rand((b, 3)) * 0.2 - 0.1
            peg_xyz[..., 2] = self.peg_half_width
            peg_q = [1, 0, 0, 0]
            peg_pose = Pose.create_from_pq(p=peg_xyz, q=peg_q)
            self.peg.set_pose(peg_pose)
            # initialize the target peg
            goal_peg_xyz = peg_xyz - torch.tensor([0, 0.2, 0])
            goal_region_q = [1, 0, 0, 0]
            goal_region_pose = Pose.create_from_pq(p=goal_peg_xyz, q=goal_region_q)
            self.goal_peg.set_pose(goal_region_pose)

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )

        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                peg_pose=self.peg.pose.raw_pose,
                goal_peg_pose=self.peg.pose.raw_pose,
                peg_to_goal_pos=self.goal_peg.pose.p - self.peg.pose.p,
                peg_to_goal_angle=info["angle_diff"],
                tcp_to_peg_pos=self.peg.pose.p - self.agent.tcp.pose.p,
                tcp_to_goal_pos=self.goal_peg.pose.p - self.agent.tcp.pose.p,
            )
        return obs

    def evaluate(self):
        right_peg_pos = (
            torch.linalg.norm(self.peg.pose.p - self.goal_peg.pose.p, axis=1) < 0.01
        )
        q = self.peg.pose.q
        qmat = rotation_conversions.quaternion_to_matrix(q)
        euler = rotation_conversions.matrix_to_euler_angles(qmat, "XYZ")
        angle_diff = torch.abs(euler[:, 2])
        is_peg_straight = angle_diff < 0.05

        is_peg_grasped = self.agent.is_grasping(self.peg)
        is_peg_static = self.peg.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        return {
            "success": right_peg_pos
            & is_peg_straight
            & is_peg_static
            & (~is_peg_grasped),
            "is_peg_placed": right_peg_pos & is_peg_straight,
            "is_peg_grasped": is_peg_grasped,
            "angle_diff": angle_diff,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reach peg
        tcp_pos = self.agent.tcp.pose.p
        tgt_tcp_pose = self.peg.pose
        tcp_to_peg_dist = torch.linalg.norm(tcp_pos - tgt_tcp_pose.p, axis=1)
        reached = tcp_to_peg_dist < 0.01
        reaching_reward = 2 * (1 - torch.tanh(5.0 * tcp_to_peg_dist))
        reward = reaching_reward
        # grasp and place reward
        angle_diff = info["angle_diff"]
        align_reward = 1 - torch.tanh(5.0 * angle_diff)
        peg_to_goal_dist = torch.linalg.norm(
            self.goal_peg.pose.p - self.peg.pose.p, axis=1
        )
        pos_reward = 1 - torch.tanh(5 * peg_to_goal_dist)
        is_peg_grasped = info["is_peg_grasped"] * reached
        reward[is_peg_grasped] = (4 + pos_reward + align_reward)[is_peg_grasped]
        # ungrasp and static reward
        gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(
            self.device
        )  # NOTE: hard-coded with panda
        ungrasp_reward = (
            torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward[~is_peg_grasped] = 1.0
        v = torch.linalg.norm(self.peg.linear_velocity, axis=1)
        av = torch.linalg.norm(self.peg.angular_velocity, axis=1)
        static_reward = 1 - torch.tanh(v * 10 + av)
        reward[info["is_peg_placed"]] = (7 + ungrasp_reward + static_reward)[
            info["is_peg_placed"]
        ]

        reward[info["success"]] = 10
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 10.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
