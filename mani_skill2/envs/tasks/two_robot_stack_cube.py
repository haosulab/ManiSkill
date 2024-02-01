from collections import OrderedDict
from typing import Any, Dict

import numpy as np
import torch

from mani_skill2.agents.multi_agent import MultiAgent
from mani_skill2.agents.robots.panda import Panda
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.envs.utils.randomization.pose import random_quaternions
from mani_skill2.envs.utils.randomization.samplers import UniformPlacementSampler
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.building import actors
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import look_at, to_tensor
from mani_skill2.utils.scene_builder.table.table_scene_builder import TableSceneBuilder
from mani_skill2.utils.structs.pose import Pose


@register_env("TwoRobotStackCube-v1", max_episode_steps=50)
class TwoRobotStackCube(BaseEnv):
    """
    Task Description
    ----------------
    The goal is to have one robot pick up the green cube and the other robot pick up the blue cube. Then the green cube has to be placed down at
    the target region and the blue cube has to be stacked on top. Note that each robot can only reach one of the cubes to begin with so they must work together
    to solve the task efficiently.

    Randomizations
    --------------
    - both cubes have their z-axis rotation randomized
    - both cubes have their xy positions on top of the table scene randomized. The positions are sampled such that the cubes do not collide with each other and
    so that the green cube is close to the robot on the left and the blue cube is close to the robot on the right.
    - the goal region is initialized in the middle between the two robots (so its y = 0). The only randomization is that it can shift along the mid-line between the two robots

    Success Conditions
    ------------------
    - the red cube is on top of the green cube (to within half of the cube size)
    - the red cube is static
    - the red cube is not being grasped by the robot (robot must let go of the cube)

    Visualization: TODO
    """

    SUPPORTED_ROBOTS = [["panda", "panda"]]
    agent: MultiAgent
    goal_radius = 0.1

    def __init__(
        self, *args, robot_uids=["panda", "panda"], robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _register_sensors(self):
        pose = look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig("base_camera", pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10)
        ]

    def _register_human_render_cameras(self):
        pose = look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose.p, pose.q, 512, 512, 1, 0.01, 10)

    def _load_actors(self):
        self.cube_half_size = to_tensor([0.02] * 3)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cubeA = actors.build_cube(
            self._scene,
            half_size=0.02,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cubeA",
        )
        self.cubeB = actors.build_cube(
            self._scene, half_size=0.02, color=[0, 1, 0, 1], name="cubeB"
        )
        self.goal_region = actors.build_red_white_target(
            self._scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
        )

    def _initialize_actors(self):
        with torch.device(self.device):
            self.table_scene.initialize()
            # the table scene initializes two robots. the first one self.agents[0] is on the left and the second one is on the right

            xyz = torch.zeros((self.num_envs, 3))
            xyz[:, 2] = 0.02
            xy = torch.rand((self.num_envs, 2)) * 0.2 - 0.1
            region = [[-0.1, -0.2], [0.1, 0.2]]
            sampler = UniformPlacementSampler(bounds=region, batch_size=self.num_envs)
            radius = (torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001).to(
                self.device
            )
            cubeA_xy = xy + sampler.sample(radius, 100)
            cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)

            xyz[:, :2] = cubeA_xy
            qs = random_quaternions(
                self.num_envs,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

            xyz[:, :2] = cubeB_xy
            qs = random_quaternions(
                self.num_envs,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            self.cubeB.set_pose(Pose.create_from_pq(p=xyz, q=qs))

    @property
    def left_agent(self) -> Panda:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> Panda:
        return self.agent.agents[1]

    def evaluate(self):
        pos_A = self.cubeA.pose.p
        pos_B = self.cubeB.pose.p
        offset = pos_A - pos_B
        xy_flag = (
            torch.linalg.norm(offset[..., :2], axis=1)
            <= torch.linalg.norm(self.cube_half_size[:2]) + 0.005
        )
        z_flag = torch.abs(offset[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.005
        is_cubeA_on_cubeB = torch.logical_and(xy_flag, z_flag)
        # TODO (stao): GPU sim can be fast but unstable. Angular velocity is rather high despite it not really rotating
        is_cubeA_static = self.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_cubeA_grasped = self.left_agent.is_grasping(self.cubeA)
        success = is_cubeA_on_cubeB * is_cubeA_static * (~is_cubeA_grasped)
        return {
            "is_cubeA_grasped": is_cubeA_grasped,
            "is_cubeA_on_cubeB": is_cubeA_on_cubeB,
            "is_cubeA_static": is_cubeA_static,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: Dict):
        obs = OrderedDict(tcp_pose=self.left_agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                # cubeA_pose=self.cubeA.pose.raw_pose,
                # cubeB_pose=self.cubeB.pose.raw_pose,
                tcp_to_cubeA_pos=self.cubeA.pose.p - self.left_agent.tcp.pose.p,
                # tcp_to_cubeB_pos=self.cubeB.pose.p - self.agent.tcp.pose.p,
                cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward
        tcp_pose = self.left_agent.tcp.pose.p
        cubeA_pos = self.cubeA.pose.p
        cubeA_to_tcp_dist = torch.linalg.norm(tcp_pose - cubeA_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * cubeA_to_tcp_dist))

        # grasp and place reward
        cubeA_pos = self.cubeA.pose.p
        cubeB_pos = self.cubeB.pose.p
        goal_xyz = torch.hstack(
            [cubeB_pos[:, 0:2], (cubeB_pos[:, 2] + self.cube_half_size[2] * 2)[:, None]]
        )
        cubeA_to_goal_dist = torch.linalg.norm(goal_xyz - cubeA_pos, axis=1)
        place_reward = 1 - torch.tanh(5.0 * cubeA_to_goal_dist)

        reward[info["is_cubeA_grasped"]] = (4 + place_reward)[info["is_cubeA_grasped"]]

        # ungrasp and static reward
        gripper_width = (self.left_agent.robot.get_qlimits()[0, -1, 1] * 2).to(
            self.device
        )  # NOTE: hard-coded with panda
        is_cubeA_grasped = info["is_cubeA_grasped"]
        ungrasp_reward = (
            torch.sum(self.left_agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
        )
        ungrasp_reward[~is_cubeA_grasped] = 1.0
        v = torch.linalg.norm(self.cubeA.linear_velocity, axis=1)
        av = torch.linalg.norm(self.cubeA.angular_velocity, axis=1)
        static_reward = 1 - torch.tanh(v * 10 + av)
        reward[info["is_cubeA_on_cubeB"]] = (
            6 + (ungrasp_reward + static_reward) / 2.0
        )[info["is_cubeA_on_cubeB"]]

        reward[info["success"]] = 8

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8
