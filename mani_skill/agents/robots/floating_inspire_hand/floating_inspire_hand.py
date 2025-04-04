from copy import deepcopy

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent


@register_agent()
class FloatingInspireHand(BaseAgent):
    uid = "floating_inspire_hand"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/inspire_hand/urdf/end_effector/inspire_hand/inspire_hand_right.urdf"
    # urdf_config = dict(
    #     _materials=dict(
    #         gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
    #     ),
    #     link=dict(
    #         right_link11=dict(
    #             material="gripper", patch_radius=0.05, min_patch_radius=0.04
    #         ),
    #     ),
    # )
    # disable_self_collisions = True

    root_joint_names = [
        "root_x_axis_joint",
        "root_y_axis_joint",
        "root_z_axis_joint",
        "root_x_rot_joint",
        "root_y_rot_joint",
        "root_z_rot_joint",
    ]

    @property
    def _controller_configs(self):
        hand_joint_pos = PDJointPosControllerConfig(
            joint_names=[
                "right_hand_wrist_pitch_joint",
                "right_hand_wrist_yaw_joint",
                "right_hand_thumb_CMC_yaw_joint",
            ],
            lower=None,
            upper=None,
            stiffness=2e4,
            damping=3e2,
            force_limit=20,
            normalize_action=False,
        )
        fingers_joint_pos = PDJointPosMimicControllerConfig(
            joint_names=[
                "right_hand_thumb_MCP_joint",
                "right_hand_thumb_IP_joint",
                "right_hand_index_PIP_joint",
                "right_hand_middle_PIP_joint",
                "right_hand_ring_PIP_joint",
                "right_hand_pinky_PIP_joint",
                "right_hand_thumb_CMC_pitch_joint",
                "right_hand_index_MCP_joint",
                "right_hand_middle_MCP_joint",
                "right_hand_ring_MCP_joint",
                "right_hand_pinky_MCP_joint",
            ],
            lower=None,
            upper=None,
            stiffness=2e4,
            damping=3e2,
            force_limit=20,
            mimic={
                "right_hand_thumb_MCP_joint": {
                    "joint": "right_hand_thumb_CMC_pitch_joint",
                    "multiplier": 1.3333333333333335,
                    "offset": -0.08144869842640205,
                },
                "right_hand_thumb_IP_joint": {
                    "joint": "right_hand_thumb_CMC_pitch_joint",
                    "multiplier": 0.6666666666666667,
                    "offset": -0.040724349213201026,
                },
                "right_hand_index_PIP_joint": {
                    "joint": "right_hand_index_MCP_joint",
                    "multiplier": 1.06399,
                    "offset": -0.16734800000000002,
                },
                "right_hand_middle_PIP_joint": {
                    "joint": "right_hand_middle_MCP_joint",
                    "multiplier": 1.06399,
                    "offset": -0.16734800000000002,
                },
                "right_hand_ring_PIP_joint": {
                    "joint": "right_hand_ring_MCP_joint",
                    "multiplier": 1.06399,
                    "offset": -0.16734800000000002,
                },
                "right_hand_pinky_PIP_joint": {
                    "joint": "right_hand_pinky_MCP_joint",
                    "multiplier": 1.06399,
                    "offset": -0.16734800000000002,
                },
            },
            normalize_action=False,
        )

        hand_joint_delta_pos = deepcopy(hand_joint_pos)
        hand_joint_delta_pos.use_delta = True
        hand_joint_delta_pos.normalize_action = True
        hand_joint_delta_pos.lower = -0.1
        hand_joint_delta_pos.upper = 0.1

        fingers_joint_delta_pos = deepcopy(fingers_joint_pos)
        fingers_joint_delta_pos.use_delta = True
        fingers_joint_delta_pos.normalize_action = True
        fingers_joint_delta_pos.lower = -0.1
        fingers_joint_delta_pos.upper = 0.1

        return dict(
            pd_joint_pos=dict(
                hand=hand_joint_pos,
                fingers=fingers_joint_pos,
            ),
            pd_joint_delta_pos=dict(
                hand=hand_joint_delta_pos,
                fingers=fingers_joint_delta_pos,
            ),
        )
