from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig


@register_agent()
class UnitreeH1(BaseAgent):
    uid = "unitree_h1"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/unitree_h1/urdf/h1.urdf"
    urdf_config = dict()
    fix_root_link = True

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
        body_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.body_joints,
            lower=-0.1,
            upper=0.1,
            stiffness=self.body_stiffness,
            damping=self.body_damping,
            force_limit=self.body_force_limit,
            use_delta=True,
        )
        return dict(pd_joint_delta_pos=dict(body=body_pd_joint_delta_pos))

    @property
    def _sensor_configs(self):
        return []

    def _after_init(self):
        pass
