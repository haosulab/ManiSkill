import numpy as np
import sapien

from mani_skill import ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig


@register_agent()
class UnitreeG1UpperBody(BaseAgent):
    uid = "unitree_g1_simplified_upper_body"
    urdf_path = f"{ASSET_DIR}/robots/unitree_g1/g1_simplified_upper_body.urdf"
    urdf_config = dict()
    fix_root_link = True
    load_multiple_collisions = True

    keyframes = dict(
        standing=Keyframe(
            pose=sapien.Pose(p=[0, 0, 0.755]),
            qpos=np.array([0.0] * (24)) * 1,
        )
    )

    body_joints = [
        # "left_hip_pitch_joint",
        # "right_hip_pitch_joint",
        # "torso_joint",
        # "left_hip_roll_joint",
        # "right_hip_roll_joint",
        # "left_shoulder_pitch_joint",
        # "right_shoulder_pitch_joint",
        # "left_hip_yaw_joint",
        # "right_hip_yaw_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        # "left_knee_joint",
        # "right_knee_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        # "left_ankle_pitch_joint",
        # "right_ankle_pitch_joint",
        "left_elbow_pitch_joint",
        "right_elbow_pitch_joint",
        # "left_ankle_roll_joint",
        # "right_ankle_roll_joint",
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
    body_stiffness = 1e3
    body_damping = 1e2
    body_force_limit = 100

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
        return dict(
            pd_joint_pos=dict(body=body_pd_joint_pos, balance_passive_force=True),
            pd_joint_delta_pos=dict(
                body=body_pd_joint_delta_pos, balance_passive_force=True
            ),
        )

    @property
    def _sensor_configs(self):
        return []


@register_agent()
class UnitreeG1UpperBodyRightArm(BaseAgent):
    uid = "unitree_g1_simplified_upper_body_right_arm"
    urdf_path = f"{ASSET_DIR}/robots/unitree_g1/g1_simplified_upper_body.urdf"

    keyframes = dict(
        standing=Keyframe(
            pose=sapien.Pose(p=[0, 0, 0.755]),
            qpos=np.array([0.0] * (24)) * 1,
        )
    )

    body_joints = [
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint",
        "right_elbow_roll_joint",
        "right_zero_joint",
        "right_three_joint",
        "right_five_joint",
        "right_one_joint",
        "right_four_joint",
        "right_six_joint",
        "right_two_joint",
    ]
    body_stiffness = 1e3
    body_damping = 1e2
    body_force_limit = 100

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
        passive = PassiveControllerConfig(
            [
                x
                for x in list(self.robot.active_joints_map.keys())
                if x not in self.body_joints
            ],
            damping=self.body_damping,
        )
        return dict(
            pd_joint_pos=dict(
                body=body_pd_joint_pos, passive=passive, balance_passive_force=True
            ),
            pd_joint_delta_pos=dict(
                body=body_pd_joint_delta_pos,
                passive=passive,
                balance_passive_force=True,
            ),
        )

    @property
    def _sensor_configs(self):
        return []
