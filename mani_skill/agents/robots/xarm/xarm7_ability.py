from copy import deepcopy
from typing import Tuple

import numpy as np
import sapien
import sapien.physx as physx

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils


@register_agent()
class XArm7Ability(BaseAgent):
    uid = "xarm7_ability"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/xarm7/xarm7_ability_right_hand.urdf"
    urdf_config = dict(
        _materials=dict(
            front_finger=dict(
                static_friction=2.0, dynamic_friction=1.5, restitution=0.0
            )
        ),
        link=dict(
            thumnb_L2=dict(
                material="front_finger", patch_radius=0.05, min_patch_radius=0.04
            ),
            index_L2=dict(
                material="front_finger", patch_radius=0.05, min_patch_radius=0.04
            ),
            middle_L2=dict(
                material="front_finger", patch_radius=0.05, min_patch_radius=0.04
            ),
            ring_L2=dict(
                material="front_finger", patch_radius=0.05, min_patch_radius=0.04
            ),
            pinky_L2=dict(
                material="front_finger", patch_radius=0.05, min_patch_radius=0.04
            ),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    0.0,
                    -0.4,
                    0.0,
                    0.5,
                    0.0,
                    0.9,
                    -3.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            pose=sapien.Pose(p=[0, 0, 0]),
        )
    )

    def __init__(self, *args, **kwargs):
        self.arm_joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 500

        self.hand_joint_names = [
            "thumb_q1",
            "index_q1",
            "middle_q1",
            "ring_q1",
            "pinky_q1",
            "thumb_q2",
            "index_q2",
            "middle_q2",
            "ring_q2",
            "pinky_q2",
        ]
        self.hand_stiffness = 1e3
        self.hand_damping = 1e2
        self.hand_force_limit = 50

        self.ee_link_name = "base"

        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # PD ee position
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )

        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # -------------------------------------------------------------------------- #
        # Hand
        # -------------------------------------------------------------------------- #
        hand_target_delta_pos = PDJointPosControllerConfig(
            self.hand_joint_names,
            -0.1,
            0.1,
            self.hand_stiffness,
            self.hand_damping,
            self.hand_force_limit,
            use_delta=True,
        )
        hand_target_delta_pos.use_target = True

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=hand_target_delta_pos
            ),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=hand_target_delta_pos),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose, gripper=hand_target_delta_pos
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose, gripper=hand_target_delta_pos
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def _after_init(self):
        hand_front_link_names = [
            "thumb_L2",
            "index_L2",
            "middle_L2",
            "ring_L2",
            "pinky_L2",
        ]
        self.hand_front_links = sapien_utils.get_objs_by_names(
            self.robot.get_links(), hand_front_link_names
        )

        finger_tip_link_names = [
            "thumb_tip",
            "index_tip",
            "middle_tip",
            "ring_tip",
            "pinky_tip",
        ]
        self.finger_tip_links = sapien_utils.get_objs_by_names(
            self.robot.get_links(), finger_tip_link_names
        )

        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

        self.queries: dict[str, Tuple[physx.PhysxGpuContactQuery, Tuple[int]]] = dict()
