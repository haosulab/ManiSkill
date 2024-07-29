import numpy as np
import sapien

from mani_skill import ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig


@register_agent(asset_download_ids=["unitree_h1"])
class UnitreeH1(BaseAgent):
    uid = "unitree_h1"
    urdf_path = f"{ASSET_DIR}/robots/unitree_h1/urdf/h1.urdf"
    urdf_config = dict()
    fix_root_link = False
    load_multiple_collisions = True

    keyframes = dict(
        standing=Keyframe(
            pose=sapien.Pose(p=[0, 0, 0.975]),
            qpos=np.array(
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
            )
            * 1,
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
        """Checks if H1 is standing with a simple heuristic of checking if the torso is at a minimum height"""
        # TODO add check for rotation of torso? so that robot can't fling itself off the floor and rotate everywhere?
        return (self.robot.pose.p[:, 2] > 0.8) & (self.robot.pose.p[:, 2] < 1.2)

    def is_fallen(self):
        """Checks if H1 has fallen on the ground. Effectively checks if the torso is too low"""
        return self.robot.pose.p[:, 2] < 0.3


@register_agent(asset_download_ids=["unitree_h1"])
class UnitreeH1Simplified(UnitreeH1):
    uid = "unitree_h1_simplified"
    urdf_path = f"{ASSET_DIR}/robots/unitree_h1/urdf/h1_simplified.urdf"
