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
class FloatingAbilityHandRight(BaseAgent):
    uid = "floating_ability_hand_right"
    urdf_path = (
        f"{PACKAGE_ASSET_DIR}/robots/ability_hand/ability_hand_right_floating.urdf"
    )
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

    root_joint_names = [
        "root_x_axis_joint",
        "root_y_axis_joint",
        "root_z_axis_joint",
        "root_x_rot_joint",
        "root_y_rot_joint",
        "root_z_rot_joint",
    ]

    def __init__(self, *args, **kwargs):

        self.hand_joint_names = [
            "index_q1",
            "middle_q1",
            "ring_q1",
            "pinky_q1",
            "index_q2",
            "middle_q2",
            "ring_q2",
            "pinky_q2",
            # "thumb_q1",
            # "index_q1",
            # "middle_q1",
            # "ring_q1",
            # "pinky_q1",
            # "thumb_q2",
            # "index_q2",
            # "middle_q2",
            # "ring_q2",
            # "pinky_q2",
        ]
        self.thumb_joint_names = ["thumb_q1", "thumb_q2"]
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
        float_pd_joint_pos = PDJointPosControllerConfig(
            joint_names=self.root_joint_names,
            lower=None,
            upper=None,
            stiffness=1e3,
            damping=1e2,
            force_limit=100,
            normalize_action=False,
        )

        # -------------------------------------------------------------------------- #
        # Hand
        # -------------------------------------------------------------------------- #
        hand_joint_pos = PDJointPosMimicControllerConfig(
            self.hand_joint_names,
            None,  # [0, 0, 0, 0, 0, 0, 0, 0, -2.0943951, 0],
            None,  # [2.0943951, 2.0943951, 2.0943951, 2.0943951, 2.65860, 2.65860, 2.65860, 2.65860, 0, 2.0943951],
            self.hand_stiffness,
            self.hand_damping,
            self.hand_force_limit,
            mimic={
                "index_q2": {
                    "joint": "index_q1",
                    "multiplier": 1.05851325,
                    "offset": 0.72349796,
                },
                "middle_q2": {
                    "joint": "middle_q1",
                    "multiplier": 1.05851325,
                    "offset": 0.72349796,
                },
                "ring_q2": {
                    "joint": "ring_q1",
                    "multiplier": 1.05851325,
                    "offset": 0.72349796,
                },
                "pinky_q2": {
                    "joint": "pinky_q1",
                    "multiplier": 1.05851325,
                    "offset": 0.72349796,
                },
            },
            normalize_action=False,
        )
        thumb_joint_pos = PDJointPosControllerConfig(
            self.thumb_joint_names,
            None,  # [0, 0, 0, 0, 0, 0, 0, 0, -2.0943951, 0],
            None,  # [2.0943951, 2.0943951, 2.0943951, 2.0943951, 2.65860, 2.65860, 2.65860, 2.65860, 0, 2.0943951],
            self.hand_stiffness,
            self.hand_damping,
            self.hand_force_limit,
            normalize_action=False,
        )
        # hand_target_delta_pos.use_target = True

        controller_configs = dict(
            pd_joint_pos=dict(
                root=float_pd_joint_pos, hand=hand_joint_pos, thumb=thumb_joint_pos
            ),
            # pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=hand_target_delta_pos),
            # pd_ee_delta_pose=dict(
            #     arm=arm_pd_ee_delta_pose, gripper=hand_target_delta_pos
            # ),
            # pd_ee_target_delta_pose=dict(
            #     arm=arm_pd_ee_target_delta_pose, gripper=hand_target_delta_pos
            # ),
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
