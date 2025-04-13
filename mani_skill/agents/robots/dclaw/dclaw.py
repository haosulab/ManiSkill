from copy import deepcopy
from typing import List

import numpy as np
import sapien
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.agents.utils import get_active_joint_indices
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs import ArticulationJoint, Link
from mani_skill.utils.structs.pose import vectorize_pose


@register_agent()
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
    keyframes = dict(
        rest=Keyframe(
            pose=sapien.Pose(p=[0, 0, 0.5], q=[0, 0, -1, 0]),
            qpos=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
    )

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

    def _after_init(self):
        self.tip_links: List[Link] = sapien_utils.get_objs_by_names(
            self.robot.get_links(), self.tip_link_names
        )
        self.root_joints: List[ArticulationJoint] = [
            self.robot.find_joint_by_name(n) for n in self.root_joint_names
        ]
        self.root_joint_indices = get_active_joint_indices(
            self.robot, self.root_joint_names
        )

    @property
    def _controller_configs(self):
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

        controller_configs = dict(
            pd_joint_delta_pos=dict(joint=joint_delta_pos),
            pd_joint_pos=dict(joint=joint_pos),
            pd_joint_target_delta_pos=dict(joint=joint_target_delta_pos),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def get_proprioception(self):
        """
        Get the proprioceptive state of the agent.
        """
        obs = super().get_proprioception()
        obs.update({"tip_poses": self.tip_poses.view(-1, len(self.tip_links) * 7)})

        return obs

    @property
    def tip_poses(self):
        """
        Get the tip pose for each of the finger, three fingers in total
        """
        tip_poses = [
            vectorize_pose(link.pose, device=self.device) for link in self.tip_links
        ]
        return torch.stack(tip_poses, dim=-2)
