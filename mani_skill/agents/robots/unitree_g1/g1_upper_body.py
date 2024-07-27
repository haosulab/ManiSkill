import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor


@register_agent(asset_download_ids=["unitree_g1"])
class UnitreeG1UpperBody(BaseAgent):
    uid = "unitree_g1_simplified_upper_body"
    urdf_path = f"{ASSET_DIR}/robots/unitree_g1/g1_simplified_upper_body.urdf"
    urdf_config = dict()
    fix_root_link = True
    load_multiple_collisions = True

    keyframes = dict(
        standing=Keyframe(
            pose=sapien.Pose(p=[0, 0, 0.755]),
            qpos=np.array([0.0] * (24)),
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
        return []  # TODO: Add sensors

    def _after_init(self):
        self.right_hand_finger_link_l_1 = self.robot.links_map["right_two_link"]
        self.right_hand_finger_link_r_1 = self.robot.links_map["right_four_link"]
        self.right_hand_finger_link_r_2 = self.robot.links_map["right_six_link"]
        self.right_tcp = self.robot.links_map["right_palm_link"]
        self.right_finger_joints = [
            "right_one_joint",
            "right_two_joint",
            "right_three_joint",
            "right_four_joint",
            "right_five_joint",
            "right_six_joint",
        ]
        self.right_finger_joint_indexes = [
            self.robot.active_joints_map[joint].active_index[0].item()
            for joint in self.right_finger_joints
        ]

        self.left_hand_finger_link_l_1 = self.robot.links_map["left_two_link"]
        self.left_hand_finger_link_r_1 = self.robot.links_map["left_four_link"]
        self.left_hand_finger_link_r_2 = self.robot.links_map["left_six_link"]
        self.left_tcp = self.robot.links_map["left_palm_link"]

        self.left_finger_joints = [
            "left_one_joint",
            "left_two_joint",
            "left_three_joint",
            "left_four_joint",
            "left_five_joint",
            "left_six_joint",
        ]
        self.left_finger_joint_indexes = [
            self.robot.active_joints_map[joint].active_index[0].item()
            for joint in self.left_finger_joints
        ]

    def right_hand_dist_to_open_grasp(self):
        """compute the distance from the current qpos to a open grasp qpos for the right hand"""
        return torch.mean(
            torch.abs(self.robot.qpos[:, self.right_finger_joint_indexes]), dim=1
        )

    def left_hand_dist_to_open_grasp(self):
        """compute the distance from the current qpos to a open grasp qpos for the left hand"""
        return torch.abs(self.robot.qpos[:, self.left_finger_joint_indexes])

    def right_hand_is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the robot is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.right_hand_finger_link_l_1, object
        )
        r_contact_forces_1 = self.scene.get_pairwise_contact_forces(
            self.right_hand_finger_link_r_1, object
        )
        r_contact_forces_2 = self.scene.get_pairwise_contact_forces(
            self.right_hand_finger_link_r_2, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce_1 = torch.linalg.norm(r_contact_forces_1, axis=1)
        rforce_2 = torch.linalg.norm(r_contact_forces_2, axis=1)

        # direction to open the gripper
        ldirection = self.right_hand_finger_link_l_1.pose.to_transformation_matrix()[
            ..., :3, 1
        ]
        rdirection1 = -self.right_hand_finger_link_r_1.pose.to_transformation_matrix()[
            ..., :3, 1
        ]
        rdirection2 = -self.right_hand_finger_link_r_2.pose.to_transformation_matrix()[
            ..., :3, 1
        ]

        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle1 = common.compute_angle_between(rdirection1, r_contact_forces_1)
        rangle2 = common.compute_angle_between(rdirection2, r_contact_forces_2)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag1 = torch.logical_and(
            rforce_1 >= min_force, torch.rad2deg(rangle1) <= max_angle
        )
        rflag2 = torch.logical_and(
            rforce_2 >= min_force, torch.rad2deg(rangle2) <= max_angle
        )
        rflag = rflag1 | rflag2
        return torch.logical_and(lflag, rflag)


@register_agent(asset_download_ids=["unitree_g1"])
class UnitreeG1UpperBodyRightArm(UnitreeG1UpperBody):
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
