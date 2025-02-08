from typing import Any, Dict, List

import numpy as np
import sapien
import torch

from mani_skill.agents.robots.anymal.anymal_c import ANYmalC
from mani_skill.agents.robots.unitree_go.unitree_go2 import UnitreeGo2Simplified
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig, SimConfig


class QuadrupedReachEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["anymal_c", "unitree_go2_simplified_locomotion"]
    agent: ANYmalC
    default_qpos: torch.Tensor

    _UNDESIRED_CONTACT_LINK_NAMES: List[str] = None

    def __init__(self, *args, robot_uids="anymal-c", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(max_rigid_contact_count=2**20),
            scene_config=SceneConfig(
                solver_position_iterations=4, solver_velocity_iterations=0
            ),
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.5, 0, 0.1], target=[1.0, 0, 0.0])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.agent.robot.links[0],
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-2.0, 1.5, 3], [1.5, 0.0, 0.5])
        return [
            CameraConfig(
                "render_camera",
                pose=pose,
                width=512,
                height=512,
                fov=1,
                near=0.01,
                far=100,
                # mount=self.agent.robot.links[0],
            )
        ]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 1]))

    def _load_scene(self, options: dict):
        self.ground = build_ground(self.scene, floor_width=400)
        self.goal = actors.build_sphere(
            self.scene,
            radius=0.2,
            color=[0, 1, 0, 1],
            name="goal",
            add_collision=False,
            body_type="kinematic",
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            keyframe = self.agent.keyframes["standing"]
            self.agent.robot.set_pose(keyframe.pose)
            self.agent.robot.set_qpos(keyframe.qpos)
            # sample random goal
            xyz = torch.zeros((b, 3))
            xyz[:, 0] = 2.5
            noise_scale = 1
            xyz[:, 0] = torch.rand(size=(b,)) * noise_scale - noise_scale / 2 + 2.5
            noise_scale = 2
            xyz[:, 1] = torch.rand(size=(b,)) * noise_scale - noise_scale / 2
            self.goal.set_pose(Pose.create_from_pq(xyz))

    def evaluate(self):
        is_fallen = self.agent.is_fallen()
        robot_to_goal_dist = torch.linalg.norm(
            self.goal.pose.p[:, :2] - self.agent.robot.pose.p[:, :2], axis=1
        )
        reached_goal = robot_to_goal_dist < 0.35
        return {
            "success": reached_goal & ~is_fallen,
            "fail": is_fallen,
            "robot_to_goal_dist": robot_to_goal_dist,
            "reached_goal": reached_goal,
            "is_fallen": is_fallen,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            root_linear_velocity=self.agent.robot.root_linear_velocity,
            root_angular_velocity=self.agent.robot.root_angular_velocity,
            reached_goal=info["success"],
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                goal_pos=self.goal.pose.p[:, :2],
                robot_to_goal=self.goal.pose.p[:, :2] - self.agent.robot.pose.p[:, :2],
            )
        return obs

    def _compute_undesired_contacts(self, threshold=1.0):
        forces = self.agent.robot.get_net_contact_forces(
            self._UNDESIRED_CONTACT_LINK_NAMES
        )
        contact_exists = torch.norm(forces, dim=-1).max(-1).values > threshold
        return contact_exists

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        robot_to_goal_dist = info["robot_to_goal_dist"]
        reaching_reward = 1 - torch.tanh(1 * robot_to_goal_dist)

        # various penalties:
        lin_vel_z_l2 = torch.square(self.agent.robot.root_linear_velocity[:, 2])
        ang_vel_xy_l2 = (
            torch.square(self.agent.robot.root_angular_velocity[:, :2])
        ).sum(axis=1)
        penalties = (
            lin_vel_z_l2 * -2
            + ang_vel_xy_l2 * -0.05
            + self._compute_undesired_contacts() * -1
            + torch.linalg.norm(self.agent.robot.qpos - self.default_qpos, axis=1)
            * -0.05
        )
        reward = 1 + 2 * reaching_reward + penalties
        reward[info["fail"]] = 0
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 3.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward


@register_env("AnymalC-Reach-v1", max_episode_steps=200)
class AnymalCReachEnv(QuadrupedReachEnv):
    """
    **Task Description:**
    Control the AnymalC robot to reach a target location in front of it. Note the current reward function works but more needs to be added to constrain the learned quadruped gait looks more natural

    **Randomizations:**
    - Robot is initialized in a stable rest/standing position
    - The goal for the robot to reach is initialized 2.5 +/- 0.5 meters in front, and +/- 1 meters to either side

    **Success Conditions:**
    - If the robot position is within 0.35 meters of the goal

    **Fail Conditions:**
    - If the robot has fallen over, which is considered True when the main body (the center part) hits the ground

    **Goal Specification:**
    - The 2D goal position in the XY-plane
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/AnymalC-Reach-v1_rt.mp4"
    _UNDESIRED_CONTACT_LINK_NAMES = ["LF_KFE", "RF_KFE", "LH_KFE", "RH_KFE"]

    def __init__(self, *args, robot_uids="anymal_c", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        self.default_qpos = torch.from_numpy(ANYmalC.keyframes["standing"].qpos).to(
            self.device
        )


@register_env("UnitreeGo2-Reach-v1", max_episode_steps=200)
class UnitreeGo2ReachEnv(QuadrupedReachEnv):
    _UNDESIRED_CONTACT_LINK_NAMES = ["FR_thigh", "RR_thigh", "FL_thigh", "RL_thigh"]

    def __init__(self, *args, robot_uids="unitree_go2_simplified_locomotion", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        self.default_qpos = torch.from_numpy(
            UnitreeGo2Simplified.keyframes["standing"].qpos
        ).to(self.device)
