from collections import OrderedDict
from typing import Dict, Union

import numpy as np
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Fetch, Panda, Xmate3Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.scene_builder.table.table_scene_builder import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array


@register_env("LiftPegUpright-v1", max_episode_steps=50)
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

    cube_half_size = 0.02

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _sensor_configs(self):
        pose = look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig("base_camera", pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10)
        ]

    @property
    def _human_render_camera_configs(self):
        pose = look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose.p, pose.q, 512, 512, 1, 0.01, 10)

    def _load_scene(self):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # the peg that we want to manipulate
        self.obj = actors.build_twocolor_peg(
            self._scene,
            length=0.12,
            width=0.025,
            color_1=np.array([200, 42, 160, 255]) / 255,
            color_2=np.array([12, 42, 160, 255]) / 255,
            name="peg",
            body_type="dynamic",
        )

    def _initialize_episode(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize()

            xyz = torch.zeros((b, 3))
            xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[..., 2] = self.cube_half_size
            q = euler2quat(np.pi / 2, 0, 0)

            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.obj.set_pose(obj_pose)

    def evaluate(self):
        # success is achieved when the peg's orientation on the table is close to upright
        # TODO: Fix for batches
        quat = np.array(self.obj.pose.q[..., :][0])
        upright1 = np.array(euler2quat(np.pi / 2, np.pi / 2, 0))
        upright2 = np.array(euler2quat(np.pi / 2, -np.pi / 2, 0))
        is_obj_upright = torch.tensor(
            np.allclose(quat, upright1, rtol=0, atol=0.005)
            or np.allclose(quat, upright2, rtol=0, atol=0.005)
        )
        return {
            "success": is_obj_upright,
        }

    def _get_obs_extra(self, info: Dict):

        obs = OrderedDict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_region.pose.p,
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                obj_pose=self.obj.pose.raw_pose,
            )
        return obs
