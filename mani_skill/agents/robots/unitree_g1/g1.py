import numpy as np
import sapien

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig


@register_agent()
class UnitreeG1(BaseAgent):
    uid = "unitree_g1"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/g1_humanoid/g1.urdf"
    urdf_config = dict()
    fix_root_link = False
    load_multiple_collisions = True

    keyframes = dict(
        standing=Keyframe(
            pose=sapien.Pose(p=[0, 0, 0.755]),
            # fmt: off
            qpos=np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, -0.77, -0.77, 0.0, 0.77, 0.77, 0.1, -0.92, -0.92, -0.1, 0.92, 0.92, 0.92, -0.92]
            ),
        ),
        right_knee_up=Keyframe(
            pose=sapien.Pose(p=[0, 0, 0.755]),
            # fmt: off
            qpos=np.array(
                [0.0, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, -0.2, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, -0.77, -0.77, 0.0, 0.77, 0.77, 0.1, -0.92, -0.92, -0.1, 0.92, 0.92, 0.92, -0.92]
            ),
        ),
        left_knee_up=Keyframe(
            pose=sapien.Pose(p=[0, 0, 0.755]),
            # fmt: off
            qpos=np.array(
                [-1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, -0.2, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, -0.77, -0.77, 0.0, 0.77, 0.77, 0.1, -0.92, -0.92, -0.1, 0.92, 0.92, 0.92, -0.92]
            ),
        ),
    )

    body_joints = [
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "torso_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_elbow_pitch_joint",
        "right_elbow_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "left_elbow_roll_joint",
        "right_elbow_roll_joint",
        "left_zero_joint",
        "left_three_joint",
        "left_five_joint",
        "right_zero_joint",
        "right_three_joint",
        "right_five_joint",
        "left_one_joint",
        "left_four_joint",
        "left_six_joint",
        "right_one_joint",
        "right_four_joint",
        "right_six_joint",
        "left_two_joint",
        "right_two_joint",
    ]
    body_stiffness = 50
    body_damping = 1
    body_force_limit = 100

    lower_body_joints = [
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
    ]
    upper_body_joints = [
        "torso_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_elbow_pitch_joint",
        "right_elbow_pitch_joint",
        "left_elbow_roll_joint",
        "right_elbow_roll_joint",
    ]
    left_hand_joints = [
        "left_zero_joint",
        "left_three_joint",
        "left_five_joint",
        "left_one_joint",
        "left_four_joint",
        "left_six_joint",
        "left_two_joint",
    ]
    right_hand_joints = [
        "right_zero_joint",
        "right_three_joint",
        "right_five_joint",
        "right_one_joint",
        "right_four_joint",
        "right_six_joint",
        "right_two_joint",
    ]

    @property
    def _controller_configs(self):
        body_pd_joint_pos = PDJointPosControllerConfig(
            self.body_joints,
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
        # note we must add balance_passive_force=False otherwise gravity will be disabled for the robot itself
        # balance_passive_force=True is only useful for fixed robots
        return dict(
            pd_joint_pos=dict(body=body_pd_joint_pos, balance_passive_force=False),
            pd_joint_delta_pos=dict(
                body=body_pd_joint_delta_pos, balance_passive_force=False
            ),
        )

    @property
    def _sensor_configs(self):
        return []

    def is_standing(self):
        """Checks if G1 is standing with a simple heuristic of checking if the torso is at a minimum height"""
        # TODO add check for rotation of torso? so that robot can't fling itself off the floor and rotate everywhere?
        return (self.robot.pose.p[:, 2] > 0.5) & (self.robot.pose.p[:, 2] < 1.0)

    def is_fallen(self):
        """Checks if G1 has fallen on the ground. Effectively checks if the torso is too low"""
        return self.robot.pose.p[:, 2] < 0.3


@register_agent()
class UnitreeG1Simplified(UnitreeG1):
    uid = "unitree_g1_simplified_legs"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/g1_humanoid/g1_simplified_legs.urdf"
