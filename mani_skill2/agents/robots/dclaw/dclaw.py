from copy import deepcopy
from typing import List

import sapien
import torch

from mani_skill2 import PACKAGE_ASSET_DIR
from mani_skill2.agents.base_agent import BaseAgent
from mani_skill2.agents.controllers import *
from mani_skill2.agents.utils import (
    get_active_joint_indices,
)
from mani_skill2.utils.sapien_utils import (
    get_objs_by_names,
)
from mani_skill2.utils.structs.pose import vectorize_pose


class DClaw(BaseAgent):
    uid = "dclaw"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/dclaw/dclaw_gripper_glb.urdf"
    urdf_config = dict(
        _materials=dict(
            tip=dict(static_friction=2.0, dynamic_friction=1.0, restitution=0.0)
        ),
        link=dict(
            link_f1_3=dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
            link_f2_3=dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
            link_f3_3=dict(material="tip", patch_radius=0.1, min_patch_radius=0.1),
        ),
    )
    sensor_configs = {}

    def __init__(self, *args, **kwargs):
        self.joint_names = [
            "joint_f1_0",
            "joint_f2_0",
            "joint_f3_0",
            "joint_f1_1",
            "joint_f2_1",
            "joint_f3_1",
            "joint_f1_2",
            "joint_f2_2",
            "joint_f3_2",
        ]

        self.joint_stiffness = 1e2
        self.joint_damping = 1e1
        self.joint_force_limit = 2e1
        self.tip_link_names = ["link_f1_head", "link_f2_head", "link_f3_head"]
        self.root_joint_names = ["joint_f1_0", "joint_f2_0", "joint_f3_0"]

        super().__init__(*args, **kwargs)
        self.tip_poses = torch.zeros(1).to(self.device)

    def _after_init(self):
        self.tip_links: List[sapien.Entity] = get_objs_by_names(
            self.robot.get_links(), self.tip_link_names
        )
        self.root_joints = [
            self.robot.find_joint_by_name(n) for n in self.root_joint_names
        ]
        self.root_joint_indices = get_active_joint_indices(
            self.robot, self.root_joint_names
        )

    @property
    def controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        joint_pos = PDJointPosControllerConfig(
            self.joint_names,
            None,
            None,
            self.joint_stiffness,
            self.joint_damping,
            self.joint_force_limit,
            normalize_action=False,
        )
        joint_delta_pos = PDJointPosControllerConfig(
            self.joint_names,
            -0.1,
            0.1,
            self.joint_stiffness,
            self.joint_damping,
            self.joint_force_limit,
            use_delta=True,
        )
        joint_target_delta_pos = deepcopy(joint_delta_pos)
        joint_target_delta_pos.use_target = True

        # PD joint velocity
        pd_joint_vel = PDJointVelControllerConfig(
            self.joint_names,
            -1.0,
            1.0,
            self.joint_damping,  # this might need to be tuned separately
            self.joint_force_limit,
        )

        # PD joint position and velocity
        joint_pos_vel = PDJointPosVelControllerConfig(
            self.joint_names,
            None,
            None,
            self.joint_stiffness,
            self.joint_damping,
            self.joint_force_limit,
            normalize_action=False,
        )
        joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.joint_names,
            -0.1,
            0.1,
            self.joint_stiffness,
            self.joint_damping,
            self.joint_force_limit,
            use_delta=True,
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(joint=joint_delta_pos),
            pd_joint_pos=dict(joint=joint_pos),
            pd_joint_target_delta_pos=dict(joint=joint_target_delta_pos),
            # Caution to use the following controllers
            pd_joint_vel=dict(joint=pd_joint_vel),
            pd_joint_pos_vel=dict(joint=joint_pos_vel),
            pd_joint_delta_pos_vel=dict(joint=joint_delta_pos_vel),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def get_proprioception(self):
        """
        Get the proprioceptive state of the agent.
        """
        obs = super().get_proprioception()
        tip_poses = [vectorize_pose(link.pose) for link in self.tip_links]
        self.tip_poses = torch.stack(tip_poses, dim=1)
        obs.update({"tip_poses": self.tip_poses.view(-1, 21)})

        return obs