import numpy as np
import sapien
import torch
from lxml import etree as ET

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils.geometry import rotation_conversions


@register_agent()
class UnitreeG1LowerBody(BaseAgent):
    uid = "unitree_g1_lower_body"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/g1_humanoid/g1_12dof.urdf"
    urdf_config = dict()
    fix_root_link = False
    load_multiple_collisions = True

    body_joints = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ]

    default_joint_angles = {  # = target angles [rad] when action = 0.0
        "left_hip_roll_joint": 0,
        "left_hip_pitch_joint": -0.1,
        "left_hip_yaw_joint": 0.0,
        "left_knee_joint": 0.3,
        "left_ankle_roll_joint": 0,
        "left_ankle_pitch_joint": -0.2,
        "right_hip_roll_joint": 0,
        "right_hip_pitch_joint": -0.1,
        "right_hip_yaw_joint": 0.0,
        "right_knee_joint": 0.3,
        "right_ankle_roll_joint": 0,
        "right_ankle_pitch_joint": -0.2,
    }

    joint_to_stiffness = {
        "left_hip_roll_joint": 100,  # hip_roll value
        "left_hip_pitch_joint": 100,  # hip_pitch value
        "left_hip_yaw_joint": 100,  # hip_yaw value
        "left_knee_joint": 150,  # knee value
        "left_ankle_roll_joint": 40,  # ankle value
        "left_ankle_pitch_joint": 40,  # ankle value
        "right_hip_roll_joint": 100,  # hip_roll value
        "right_hip_pitch_joint": 100,  # hip_pitch value
        "right_hip_yaw_joint": 100,  # hip_yaw value
        "right_knee_joint": 150,  # knee value
        "right_ankle_roll_joint": 40,  # ankle value
        "right_ankle_pitch_joint": 40,  # ankle value
    }  # [N*m/rad]
    joint_to_damping = {
        "left_hip_roll_joint": 2,  # hip_roll value
        "left_hip_pitch_joint": 2,  # hip_pitch value
        "left_hip_yaw_joint": 2,  # hip_yaw value
        "left_knee_joint": 4,  # knee value
        "left_ankle_roll_joint": 2,  # ankle value
        "left_ankle_pitch_joint": 2,  # ankle value
        "right_hip_roll_joint": 2,  # hip_roll value
        "right_hip_pitch_joint": 2,  # hip_pitch value
        "right_hip_yaw_joint": 2,  # hip_yaw value
        "right_knee_joint": 4,  # knee value
        "right_ankle_roll_joint": 2,  # ankle value
        "right_ankle_pitch_joint": 2,  # ankle value
    }  # [N*m*s/rad]

    def __init__(self, *args, **kwargs):
        with open(self.urdf_path, "r") as f:
            urdf_string = f.read()
        xml = ET.fromstring(urdf_string.encode("utf-8"))

        joint_name_to_limits = dict()
        for node in xml:
            if node.tag == "joint":
                for child in node:
                    if child.tag == "limit":
                        joint_name_to_limits[node.get("name")] = dict(
                            (k, float(v)) for k, v in child.items()
                        )
        self.joint_to_vel_limit = dict(
            (jname, joint_name_to_limits[jname]["velocity"])
            for jname in self.body_joints
        )
        self.joint_to_torque_limit = dict(
            (jname, joint_name_to_limits[jname]["effort"]) for jname in self.body_joints
        )

        self.body_stiffness = [
            self.joint_to_stiffness[joint] for joint in self.body_joints
        ]
        self.body_damping = [self.joint_to_damping[joint] for joint in self.body_joints]
        self.body_force_limit = [
            self.joint_to_torque_limit[joint] for joint in self.body_joints
        ]

        super().__init__(*args, **kwargs)

    def _after_init(self):
        self.keyframes = dict(
            standing=Keyframe(
                pose=sapien.Pose(p=[0, 0, 0.8]),
                qpos=np.array(
                    [
                        self.default_joint_angles[j.name]
                        for j in self.robot.active_joints
                    ]
                ),
            ),
        )
        self.terminate_after_contacts_on_link_names = [
            l for l in self.robot_link_names if "pelvis" in self.robot_link_names
        ]
        self.torso = self.robot.find_link_by_name("torso_link")

    @property
    def _controller_configs(self):
        body_pd_joint_pos = PDJointPosControllerConfig(
            self.body_joints,
            lower=None,
            upper=None,
            stiffness=self.body_stiffness,
            damping=self.body_damping,
            force_limit=self.body_force_limit,
            use_delta=False,
            normalize_action=False,
        )
        # note we must add balance_passive_force=False otherwise gravity will be disabled for the robot itself
        # balance_passive_force=True is only useful for fixed robots
        return dict(
            pd_joint_pos=dict(body=body_pd_joint_pos, balance_passive_force=False)
        )

    @property
    def _sensor_configs(self):
        return []

    def is_standing(self):
        """Checks if G1 is standing; defined as roll/pitch within limit and pelvis has no contacts"""
        return ~self.is_fallen()

    def is_fallen(self):
        """Checks if G1 is fallen; defined as roll/pitch exceeds limit or pelvis has contact"""
        rpy = rotation_conversions.quaternion_to_euler_angles(self.robot.pose.q, "XYZ")
        return (
            (rpy[:, 0].abs() > 0.8)
            | (rpy[:, 1].abs() > 1.0)
            # | (
            #     self.robot.get_net_contact_forces(
            #         self.terminate_after_contacts_on_link_names
            #     )
            #     > 1.0
            # )
            # .any(dim=-1)
            # .any(dim=-1)
        )


@register_agent()
class UnitreeG1LowerBodySimplified(BaseAgent):
    uid = "unitree_g1_lower_body_simplified"
    urdf_path = (
        f"{PACKAGE_ASSET_DIR}/robots/g1_humanoid/g1_12dof_simplified_collisions.urdf"
    )
