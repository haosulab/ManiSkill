import os.path as osp
from collections import OrderedDict
from typing import Any, Dict, Union

import numpy as np
import torch
import torch.random
from transforms3d.euler import euler2quat, quat2euler
import sapien

from mani_skill2.agents.robots.fetch.fetch import Fetch
from mani_skill2.agents.robots.panda.panda import Panda
from mani_skill2.agents.robots.xmate3.xmate3 import Xmate3Robotiq
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.building import actors
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import look_at
from mani_skill2.utils.scene_builder.table.table_scene_builder import TableSceneBuilder
from mani_skill2.utils.structs.pose import Pose
from mani_skill2.utils.structs.types import Array, GPUMemoryConfig, SimConfig


@register_env("LiftPegUpright-v0", max_episode_steps=50)
class LiftPegUprightEnv(BaseEnv):
    """
    Task Description
    ----------------
    A simple task where the objective is to move a peg laying on the table to the upright position

    Randomizations
    --------------
    - the peg's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat along it's length on the table

    Success Conditions
    ------------------
    - the peg's orientation (given by the quaternion) is within close to the upright orientation ([np.pi/2, np.pi/2, 0] or [np.pi/2, -np.pi/2, 0])

    Visualization: TODO: ADD LINK HERE
    """

    SUPPORTED_ROBOTS = ["panda", "xmate3_robotiq", "fetch"]
    agent: Union[Panda, Xmate3Robotiq, Fetch]

    # Specify default simulation/gpu memory configurations.
    sim_cfg = SimConfig(
        gpu_memory_cfg=GPUMemoryConfig(
            found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
        )
    )

    # set some commonly used values
    cube_half_size = 0.02

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
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
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # the peg that we want to manipulate
        self.obj = actors.build_twocolor_peg(
            self._scene,
            length=0.12,
            width=0.025,
            color_1=np.array([12, 42, 160, 255]) / 255,
            color_2=np.array([12, 42, 160, 255]) / 255,
            name="peg",
            body_type="dynamic",
        )

        builder = self._scene.create_actor_builder()

    def _initialize_actors(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize()

            xyz = torch.zeros((b, 3))
            xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[..., 2] = self.cube_half_size
            q = euler2quat(np.pi/2, 0, 0)

            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.obj.set_pose(obj_pose)

    def evaluate(self):
        # success is achieved when the peg's orientation on the table is close to upright
        # TODO: Fix for batches
        quat = np.array(self.obj.pose.q[... , :][0])
        upright1 = np.array(euler2quat(np.pi/2, np.pi/2, 0))
        upright2 = np.array(euler2quat(np.pi/2, -np.pi/2, 0))
        is_obj_upright = torch.tensor(
            np.allclose(quat, upright1, rtol=0, atol=0.005) or np.allclose(quat, upright2, rtol=0, atol=0.005)
        )

        return {
            "success": is_obj_upright,
        }

    def _get_obs_extra(self, info: Dict):
        # some useful observation info for solving the task includes the pose of the tcp (tool center point) which is the point between the
        # grippers of the robot
        obs = OrderedDict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_region.pose.p,
        )
        if self._obs_mode in ["state", "state_dict"]:
            # if the observation mode is state/state_dict, we provide ground truth information about where the cube is.
            # for visual observation modes one should rely on the sensed visual data to determine where the cube is
            obs.update(
                obj_pose=self.obj.pose.raw_pose,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        # We also create a pose marking where the robot should push the cube from that is easiest (pushing from behind the cube)
        tcp_push_pose = Pose.create_from_pq(
            p=self.obj.pose.p
            + torch.tensor([-self.cube_half_size - 0.005, 0, 0], device=self.device)
        )
        tcp_to_push_pose = tcp_push_pose.p - self.agent.tcp.pose.p
        tcp_to_push_pose_dist = torch.linalg.norm(tcp_to_push_pose, axis=1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_push_pose_dist)
        reward = reaching_reward

        # compute a placement reward to encourage robot to move the cube to the center of the goal region
        # we further multiply the place_reward by a mask reached so we only add the place reward if the robot has reached the desired push pose
        # This reward design helps train RL agents faster by staging the reward out.
        reached = tcp_to_push_pose_dist < 0.01
        obj_to_goal_dist = torch.linalg.norm(
            self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * reached

        # assign rewards to parallel environments that achieved success to the maximum of 3.
        reward[info["success"]] = 3
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 3.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
