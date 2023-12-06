"""
NOTE (stao): note that the existence of a dummy mobile agent file here suggests likely there is a lot of code for mobile agents we can abstract away / share
and avoid using a base_class.py kind of file
"""
from typing import Sequence, TypeVar, Union

import numpy as np
from sapien import Pose
from transforms3d.euler import euler2quat

from mani_skill2.agents.base_agent import BaseAgent
from mani_skill2.agents.controllers import *
from mani_skill2.utils.sapien_utils import get_obj_by_name


class DummyMobileAgent(BaseAgent):
    def __init__(self, scene, control_freq, control_mode=None, fix_root_link=True):
        if control_mode is None:  # if user did not specify a control_mode
            control_mode = "base_pd_joint_vel_arm_pd_joint_vel"
        super().__init__(
            scene,
            control_freq,
            control_mode=control_mode,
            fix_root_link=fix_root_link,
        )

    def _after_init(self):
        super()._after_init()

        # Sanity check
        active_joints = self.robot.get_active_joints()
        assert active_joints[0].name == "root_x_axis_joint"
        assert active_joints[1].name == "root_y_axis_joint"
        assert active_joints[2].name == "root_z_rotation_joint"

        # Dummy base
        self.base_link = self.robot.get_links()[3]

        # Ignore collision between the adjustable body and ground
        body = get_obj_by_name(self.robot.get_links(), "adjustable_body")

        s = body.get_collision_shapes()[0]
        gs = s.get_collision_groups()
        gs[2] = gs[2] | 1 << 30
        s.set_collision_groups(gs)

    def get_proprioception(self):
        state_dict = super().get_proprioception()
        qpos, qvel = state_dict["qpos"], state_dict["qvel"]
        base_pos, base_orientation, arm_qpos = qpos[:2], qpos[2], qpos[3:]
        base_vel, base_ang_vel, arm_qvel = qvel[:2], qvel[2], qvel[3:]

        state_dict["qpos"] = arm_qpos
        state_dict["qvel"] = arm_qvel
        state_dict["base_pos"] = base_pos
        state_dict["base_orientation"] = base_orientation
        state_dict["base_vel"] = base_vel
        state_dict["base_ang_vel"] = base_ang_vel
        return state_dict

    @property
    def base_pose(self):
        qpos = self.robot.get_qpos()
        x, y, ori = qpos[:3]
        return Pose([x, y, 0], euler2quat(0, 0, ori))

    def set_base_pose(self, xy, ori):
        qpos = self.robot.get_qpos()
        qpos[0:2] = xy
        qpos[2] = ori
        self.robot.set_qpos(qpos)
