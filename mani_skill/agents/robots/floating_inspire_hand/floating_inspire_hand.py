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
    disable_self_collisions = True

    finger_joint = []

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
            stiffness=1e3,
            damping=1e2,
            force_limit=100,
        )
        mimic_joint_pos = PDJointPosMimicControllerConfig(
            joint_names=[
                "right_hand_thumb_MCP_joint",
                "right_hand_thumb_IP_joint",
                "right_hand_index_PIP_joint",
                "right_hand_middle_PIP_joint",
                "right_hand_pinky_PIP_joint",
                "right_hand_thumb_CMC_pitch_joint",
                "right_hand_index_MCP_joint",
                "right_hand_middle_MCP_joint",
                "right_hand_pinky_MCP_joint",
            ],
            lower=None,
            upper=None,
            stiffness=1e3,
            damping=1e2,
            force_limit=100,
            mimic={
                "right_hand_thumb_MCP_joint": {
                    "joint": "right_hand_thumb_CMC_pitch_joint",
                    "multiplier": 1.0,
                    "offset": 0.0,
                },
                "right_hand_thumb_IP_joint": {
                    "joint": "right_hand_thumb_CMC_pitch_joint",
                    "multiplier": 1.0,
                    "offset": 0.0,
                },
                "right_hand_index_PIP_joint": {
                    "joint": "right_hand_index_MCP_joint",
                    "multiplier": 1.0,
                    "offset": 0.0,
                },
                "right_hand_middle_PIP_joint": {
                    "joint": "right_hand_middle_MCP_joint",
                    "multiplier": 1.0,
                    "offset": 0.0,
                },
                "right_hand_pinky_PIP_joint": {
                    "joint": "right_hand_pinky_MCP_joint",
                    "multiplier": 1.0,
                    "offset": 0.0,
                },
            },
        )
        return dict(pd_joint_pos=dict(hand=hand_joint_pos, mimic=mimic_joint_pos))
