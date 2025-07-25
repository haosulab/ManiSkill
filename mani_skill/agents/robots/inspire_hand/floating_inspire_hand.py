from copy import deepcopy

import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent


@register_agent()
class FloatingInspireHandRight(BaseAgent):
    uid = "floating_inspire_hand_right"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/inspire_hand/RH56DFX-2LR/urdf/inspire_hand_right_floating.urdf"
    urdf_config = dict(
        _materials=dict(
            finger=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            right_hand_hand_base_link=dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1
            ),
            right_hand_thumb_distal=dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1
            ),
            right_hand_thumb_metacarpal=dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1
            ),
            right_hand_thumb_proximal=dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1
            ),
            right_hand_index_proximal=dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1
            ),
            right_hand_index_middle=dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1
            ),
            right_hand_middle_proximal=dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1
            ),
            right_hand_middle_middle=dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1
            ),
            right_hand_ring_proximal=dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1
            ),
            right_hand_ring_middle=dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1
            ),
            right_hand_pinky_proximal=dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1
            ),
            right_hand_pinky_middle=dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )
    disable_self_collisions = True
    # you could model all of the fingers and disable certain impossible self collisions that occur
    # but it is simpler and faster to just disable all self collisions. It is highly unlikely the hand self-collides to begin with
    # due to the design of the hand

    root_joint_names = [
        "root_x_axis_joint",
        "root_y_axis_joint",
        "root_z_axis_joint",
        "root_x_rot_joint",
        "root_y_rot_joint",
        "root_z_rot_joint",
    ]

    keyframes = dict(
        palm_side=Keyframe(
            # magic numbers below correspond with controlling joints being set to 0. The other non-active revolute joints are mimic joints
            qpos=[
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    -0.16734816,
                    -0.16734803,
                    -0.16734798,
                    -0.167348,
                    -0.08147363,
                    -0.07234851,
                ]
            ],
            pose=sapien.Pose(p=[0, 0, 0.4], q=euler2quat(0, 0, -np.pi / 2)),
        ),
        palm_up=Keyframe(
            qpos=[
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    -0.16734816,
                    -0.16734803,
                    -0.16734798,
                    -0.167348,
                    -0.08147363,
                    -0.07234851,
                ]
            ],
            pose=sapien.Pose(p=[0, 0, 0.4], q=euler2quat(np.pi / 2, 0, -np.pi / 2)),
        ),
    )

    @property
    def _controller_configs(self):
        float_pd_joint_pos = PDJointPosControllerConfig(
            joint_names=self.root_joint_names,
            lower=None,
            upper=None,
            stiffness=1e3,
            damping=1e2,
            force_limit=100,
            normalize_action=False,
        )
        wrist_joint_pos = PDJointPosControllerConfig(
            joint_names=["right_hand_wrist_pitch_joint", "right_hand_wrist_yaw_joint"],
            lower=None,
            upper=None,
            stiffness=1e3,
            damping=1e2,
            force_limit=100,
            normalize_action=False,
        )
        fingers_joint_pos = PDJointPosControllerConfig(
            joint_names=[
                "right_hand_thumb_CMC_yaw_joint",
                "right_hand_thumb_CMC_pitch_joint",
                "right_hand_index_MCP_joint",
                "right_hand_middle_MCP_joint",
                "right_hand_ring_MCP_joint",
                "right_hand_pinky_MCP_joint",
            ],
            lower=None,
            upper=None,
            stiffness=1e3,
            damping=1e2,
            force_limit=20,
            normalize_action=False,
        )
        passive = PassiveControllerConfig(
            joint_names=[
                "right_hand_thumb_MCP_joint",
                "right_hand_thumb_IP_joint",
                "right_hand_index_PIP_joint",
                "right_hand_middle_PIP_joint",
                "right_hand_ring_PIP_joint",
                "right_hand_pinky_PIP_joint",
            ],
            damping=0.001,  # NOTE (stao): This magic number is necessary for physx to simulate correctly...
            force_limit=20,
        )

        wrist_joint_delta_pos = deepcopy(wrist_joint_pos)
        wrist_joint_delta_pos.use_delta = True
        wrist_joint_delta_pos.normalize_action = True
        wrist_joint_delta_pos.lower = -0.1
        wrist_joint_delta_pos.upper = 0.1

        fingers_joint_delta_pos = deepcopy(fingers_joint_pos)
        fingers_joint_delta_pos.use_delta = True
        fingers_joint_delta_pos.normalize_action = True
        fingers_joint_delta_pos.lower = -0.1
        fingers_joint_delta_pos.upper = 0.1

        float_pd_joint_delta_pos = deepcopy(float_pd_joint_pos)
        float_pd_joint_delta_pos.use_delta = True
        float_pd_joint_delta_pos.normalize_action = True
        float_pd_joint_delta_pos.lower = -0.1
        float_pd_joint_delta_pos.upper = 0.1

        return dict(
            pd_joint_pos=dict(
                root=float_pd_joint_pos,
                wrist=wrist_joint_pos,
                fingers=fingers_joint_pos,
                passive=passive,
            ),
            pd_joint_delta_pos=dict(
                root=float_pd_joint_delta_pos,
                wrist=wrist_joint_delta_pos,
                fingers=fingers_joint_delta_pos,
                passive=passive,
            ),
        )


@register_agent()
class FloatingInspireHandLeft(BaseAgent):
    uid = "floating_inspire_hand_left"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/inspire_hand/RH56DFX-2LR/urdf/inspire_hand_left_floating.urdf"
    urdf_config = dict(
        _materials=dict(
            finger=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            left_hand_hand_base_link=dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1
            ),
            left_hand_thumb_distal=dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1
            ),
            left_hand_thumb_metacarpal=dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1
            ),
            left_hand_thumb_proximal=dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1
            ),
            left_hand_index_proximal=dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1
            ),
            left_hand_index_middle=dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1
            ),
            left_hand_middle_proximal=dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1
            ),
            left_hand_middle_middle=dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1
            ),
            left_hand_ring_proximal=dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1
            ),
            left_hand_ring_middle=dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1
            ),
            left_hand_pinky_proximal=dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1
            ),
            left_hand_pinky_middle=dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )
    disable_self_collisions = True
    # you could model all of the fingers and disable certain impossible self collisions that occur
    # but it is simpler and faster to just disable all self collisions. It is highly unlikely the hand self-collides to begin with
    # due to the design of the hand

    root_joint_names = [
        "root_x_axis_joint",
        "root_y_axis_joint",
        "root_z_axis_joint",
        "root_x_rot_joint",
        "root_y_rot_joint",
        "root_z_rot_joint",
    ]

    keyframes = dict(
        palm_side=Keyframe(
            # magic numbers below correspond with controlling joints being set to 0. The other non-active revolute joints are mimic joints
            qpos=[
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    -0.16734816,
                    -0.16734803,
                    -0.16734798,
                    -0.167348,
                    -0.08147363,
                    -0.07234851,
                ]
            ],
            pose=sapien.Pose(p=[0, 0, 0.4], q=euler2quat(0, 0, -np.pi / 2)),
        ),
        palm_up=Keyframe(
            qpos=[
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    -0.16734816,
                    -0.16734803,
                    -0.16734798,
                    -0.167348,
                    -0.08147363,
                    -0.07234851,
                ]
            ],
            pose=sapien.Pose(p=[0, 0, 0.4], q=euler2quat(np.pi / 2, 0, -np.pi / 2)),
        ),
    )

    @property
    def _controller_configs(self):
        float_pd_joint_pos = PDJointPosControllerConfig(
            joint_names=self.root_joint_names,
            lower=None,
            upper=None,
            stiffness=1e3,
            damping=1e2,
            force_limit=100,
            normalize_action=False,
        )
        wrist_joint_pos = PDJointPosControllerConfig(
            joint_names=["left_hand_wrist_pitch_joint", "left_hand_wrist_yaw_joint"],
            lower=None,
            upper=None,
            stiffness=1e3,
            damping=1e2,
            force_limit=100,
            normalize_action=False,
        )
        fingers_joint_pos = PDJointPosControllerConfig(
            joint_names=[
                "left_hand_thumb_CMC_yaw_joint",
                "left_hand_thumb_CMC_pitch_joint",
                "left_hand_index_MCP_joint",
                "left_hand_middle_MCP_joint",
                "left_hand_ring_MCP_joint",
                "left_hand_pinky_MCP_joint",
            ],
            lower=None,
            upper=None,
            stiffness=1e3,
            damping=1e2,
            force_limit=20,
            normalize_action=False,
        )
        passive = PassiveControllerConfig(
            joint_names=[
                "left_hand_thumb_MCP_joint",
                "left_hand_thumb_IP_joint",
                "left_hand_index_PIP_joint",
                "left_hand_middle_PIP_joint",
                "left_hand_ring_PIP_joint",
                "left_hand_pinky_PIP_joint",
            ],
            damping=0.001,  # NOTE (stao): This magic number is necessary for physx to simulate correctly...
            force_limit=20,
        )

        wrist_joint_delta_pos = deepcopy(wrist_joint_pos)
        wrist_joint_delta_pos.use_delta = True
        wrist_joint_delta_pos.normalize_action = True
        wrist_joint_delta_pos.lower = -0.1
        wrist_joint_delta_pos.upper = 0.1

        fingers_joint_delta_pos = deepcopy(fingers_joint_pos)
        fingers_joint_delta_pos.use_delta = True
        fingers_joint_delta_pos.normalize_action = True
        fingers_joint_delta_pos.lower = -0.1
        fingers_joint_delta_pos.upper = 0.1

        float_pd_joint_delta_pos = deepcopy(float_pd_joint_pos)
        float_pd_joint_delta_pos.use_delta = True
        float_pd_joint_delta_pos.normalize_action = True
        float_pd_joint_delta_pos.lower = -0.1
        float_pd_joint_delta_pos.upper = 0.1

        return dict(
            pd_joint_pos=dict(
                root=float_pd_joint_pos,
                wrist=wrist_joint_pos,
                fingers=fingers_joint_pos,
                passive=passive,
            ),
            pd_joint_delta_pos=dict(
                root=float_pd_joint_delta_pos,
                wrist=wrist_joint_delta_pos,
                fingers=fingers_joint_delta_pos,
                passive=passive,
            ),
        )
