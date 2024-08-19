import numpy as np
import sapien

from mani_skill import ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig

# TODO (stao): there is a bug currently with the full H1+hands robot. It breaks the simulation at the moment when all the fingers are added to the body.
# @register_agent()
# class UnitreeH1WithHands(BaseAgent):
#     uid = "unitree_h1_with_hands"
#     urdf_path = f"{ASSET_DIR}/robots/unitree_h1/urdf/h1_with_hand.urdf"
#     urdf_config = dict()
#     fix_root_link = False
#     load_multiple_collisions = True
#     disable_self_collisions = True

#     keyframes = dict(
#         standing=Keyframe(
#             pose=sapien.Pose(p=[0, 0, 0.975]),
#             qpos=np.array(
#                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4, -0.4, 0.0, 0.0, 0.8, 0.8, 0.0, 0.0, -0.4, -0.4, 0.0, 0.0, 0.0, 0.0,
#                  0.0, 0.0, 0.0, 0.0, 0.0,
#                                             #  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
#                  ]
#             )
#             * 1,
#         )
#     )
# # 0,
# #                     0,
# #                     0,
# #                     0,
# #                     0,
# #                     0,
# #                     0,
# #                     -0.4,
# #                     -0.4,
# #                     0.0,
# #                     0.0,
# #                     0.8,
# #                     0.8,
# #                     0.0,
# #                     0.0,
# #                     -0.4,
# #                     -0.4,
# #                     0.0,
# #                     0.0,
#     body_joints = [
#         "left_hip_yaw_joint",
#         "right_hip_yaw_joint",
#         "torso_joint",
#         "left_hip_roll_joint",
#         "right_hip_roll_joint",
#         "left_shoulder_pitch_joint",
#         "right_shoulder_pitch_joint",
#         "left_hip_pitch_joint",
#         "right_hip_pitch_joint",
#         "left_shoulder_roll_joint",
#         "right_shoulder_roll_joint",
#         "left_knee_joint",
#         "right_knee_joint",
#         "left_shoulder_yaw_joint",
#         "right_shoulder_yaw_joint",
#         "left_ankle_joint",
#         "right_ankle_joint",
#         "left_elbow_joint",
#         "right_elbow_joint",
#     ]
#     arm_hand_joints = [
#         'left_hand_joint', 'right_hand_joint',
#         "R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint", "R_thumb_intermediate_joint", "R_thumb_distal_joint", "R_index_proximal_joint",
#         # the following 3 joints cause sim problems "R_index_intermediate_joint", #"R_middle_proximal_joint",# "R_middle_intermediate_joint"
#     ]
#     body_stiffness = 1e4
#     body_damping = 1e3
#     body_force_limit = 10000

#     @property
#     def _controller_configs(self):
#         print(self.robot.active_joints_map.keys())
#         body_pd_joint_pos = PDJointPosControllerConfig(
#             self.body_joints,
#             lower=None,
#             upper=None,
#             stiffness=self.body_stiffness,
#             damping=self.body_damping,
#             force_limit=self.body_force_limit,
#             normalize_action=False,
#         )
#         arm_hands_pd_joint_pos = PDJointPosControllerConfig(
#             self.arm_hand_joints,
#             lower=None,
#             upper=None,
#             stiffness=self.body_stiffness,
#             damping=self.body_damping,
#             force_limit=self.body_force_limit,
#             normalize_action=False,
#         )
#         body_pd_joint_delta_pos = PDJointPosControllerConfig(
#             self.body_joints,
#             lower=-0.2,
#             upper=0.2,
#             stiffness=self.body_stiffness,
#             damping=self.body_damping,
#             force_limit=self.body_force_limit,
#             use_delta=True,
#         )
#         arm_hands_pd_joint_delta_pos = PDJointPosControllerConfig(
#             self.arm_hand_joints,
#             lower=-0.2,
#             upper=0.2,
#             stiffness=self.body_stiffness,
#             damping=self.body_damping,
#             force_limit=self.body_force_limit,
#             use_delta=True,
#         )
#         # note we must add balance_passive_force=False otherwise gravity will be disabled for the robot itself
#         # balance_passive_force=True is only useful for fixed robots
#         return dict(
#             pd_joint_pos=dict(body=body_pd_joint_pos, arm_hands=arm_hands_pd_joint_pos, balance_passive_force=False),
#             pd_joint_delta_pos=dict(
#                 body=body_pd_joint_delta_pos, arm_hands=arm_hands_pd_joint_delta_pos, balance_passive_force=False
#             ),
#         )

#     @property
#     def _sensor_configs(self):
#         return []

#     def is_standing(self):
#         """Checks if H1 is standing with a simple heuristic of checking if the torso is at a minimum height"""
#         # TODO add check for rotation of torso? so that robot can't fling itself off the floor and rotate everywhere?
#         return (self.robot.pose.p[:, 2] > 0.8) & (self.robot.pose.p[:, 2] < 1.2)

#     def is_fallen(self):
#         """Checks if H1 has fallen on the ground. Effectively checks if the torso is too low"""
#         return self.robot.pose.p[:, 2] < 0.3


@register_agent()
class UnitreeH1WithHandsUpperBodyOnly(BaseAgent):
    uid = "unitree_h1_with_hands_upper_body_only"
    urdf_path = f"{ASSET_DIR}/robots/unitree_h1/urdf/h1_with_hand.urdf"
    urdf_config = dict()
    fix_root_link = False
    load_multiple_collisions = True
    disable_self_collisions = True
    keyframes = dict(
        standing=Keyframe(
            pose=sapien.Pose(p=[0, 0, 0.975]),
            qpos=np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.4,
                    -0.4,
                    0.0,
                    0.0,
                    0.8,
                    0.8,
                    0.0,
                    0.0,
                    -0.4,
                    -0.4,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                    #  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                ]
            )
            * 1,
        )
    )
    body_joints = [
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_elbow_joint",
    ]
    arm_hand_joints = [
        "left_hand_joint",
        "right_hand_joint",
        "R_thumb_proximal_yaw_joint",
        "R_thumb_proximal_pitch_joint",
        "R_thumb_intermediate_joint",
        "R_thumb_distal_joint"
        #'L_thumb_proximal_yaw_joint', 'L_index_proximal_joint', 'L_middle_proximal_joint', 'L_ring_proximal_joint', 'L_pinky_proximal_joint', 'R_thumb_proximal_yaw_joint', 'R_index_proximal_joint', 'R_middle_proximal_joint', 'R_ring_proximal_joint', 'R_pinky_proximal_joint', 'L_thumb_proximal_pitch_joint', 'L_index_intermediate_joint', 'L_middle_intermediate_joint', 'L_ring_intermediate_joint', 'L_pinky_intermediate_joint', 'R_thumb_proximal_pitch_joint', 'R_index_intermediate_joint', 'R_middle_intermediate_joint', 'R_ring_intermediate_joint', 'R_pinky_intermediate_joint', 'L_thumb_intermediate_joint', 'R_thumb_intermediate_joint', 'L_thumb_distal_joint', 'R_thumb_distal_joint'
    ]

    @property
    def _controller_configs(self):
        print(
            self.robot.active_joints_map.keys(),
            len(self.robot.active_joints_map.keys()),
        )
        print("qposhspa", self.keyframes["standing"].qpos.shape)
        body_pd_joint_pos = PDJointPosControllerConfig(
            self.body_joints,
            lower=None,
            upper=None,
            stiffness=self.body_stiffness,
            damping=self.body_damping,
            force_limit=self.body_force_limit,
            normalize_action=False,
        )
        arm_hands_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_hand_joints,
            lower=None,
            upper=None,
            stiffness=self.body_stiffness,
            damping=self.body_damping,
            force_limit=self.body_force_limit,
            normalize_action=False,
        )
        body_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.body_joints,
            lower=-0.2,
            upper=0.2,
            stiffness=self.body_stiffness,
            damping=self.body_damping,
            force_limit=self.body_force_limit,
            use_delta=True,
        )
        arm_hands_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_hand_joints,
            lower=-0.2,
            upper=0.2,
            stiffness=self.body_stiffness,
            damping=self.body_damping,
            force_limit=self.body_force_limit,
            use_delta=True,
        )
        # note we must add balance_passive_force=False otherwise gravity will be disabled for the robot itself
        # balance_passive_force=True is only useful for fixed robots
        return dict(
            pd_joint_pos=dict(
                body=body_pd_joint_pos,
                arm_hands=arm_hands_pd_joint_pos,
                balance_passive_force=False,
            ),
            pd_joint_delta_pos=dict(
                body=body_pd_joint_delta_pos,
                arm_hands=arm_hands_pd_joint_delta_pos,
                balance_passive_force=False,
            ),
        )
