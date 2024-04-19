import numpy as np
import sapien

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig


@register_agent()
class UnitreeH1(BaseAgent):
    uid = "unitree_h1"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/unitree_h1/urdf/h1.urdf"
    urdf_config = dict()
    fix_root_link = False

    keyframes = dict(
        standing=Keyframe(
            pose=sapien.Pose(p=[0, 0, 1.03]),
            qpos=np.array(
                [
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
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
                    ]
                ]
            ),
        )
    )

    body_joints = [
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "torso_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_ankle_joint",
        "right_ankle_joint",
        "left_elbow_joint",
        "right_elbow_joint",
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
            lower=-0.1,
            upper=0.1,
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

    def _after_init(self):
        pass
