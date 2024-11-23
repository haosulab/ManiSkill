import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor


@register_agent()
class UnitreeG1UpperBody(BaseAgent):
    """The G1 Robot with control over its torso rotation and its two arms. Legs are fixed."""

    uid = "unitree_g1_simplified_upper_body"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/g1_humanoid/g1_simplified_upper_body.urdf"
    urdf_config = dict(
        _materials=dict(
            finger=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link={
            **{
                f"left_{k}_link": dict(
                    material="finger", patch_radius=0.1, min_patch_radius=0.1
                )
                for k in ["one", "two", "three", "four", "five", "six"]
            },
            **{
                f"right_{k}_link": dict(
                    material="finger", patch_radius=0.1, min_patch_radius=0.1
                )
                for k in ["one", "two", "three", "four", "five", "six"]
            },
            "left_palm_link": dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1
            ),
            "right_palm_link": dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1
            ),
        },
    )
    fix_root_link = True
    load_multiple_collisions = False

    keyframes = dict(
        standing=Keyframe(
            pose=sapien.Pose(p=[0, 0, 0.755]),
            qpos=np.array([0.0] * (25)),
        )
    )

    body_joints = [
        # "left_hip_pitch_joint",
        # "right_hip_pitch_joint",
        "torso_joint",
        # "left_hip_roll_joint",
        # "right_hip_roll_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
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
            lower=[-0.2] * 11 + [-0.5] * 14,
            upper=[0.2] * 11 + [0.5] * 14,
            stiffness=self.body_stiffness,
            damping=self.body_damping,
            force_limit=self.body_force_limit,
            use_delta=True,
        )
        return dict(
            pd_joint_delta_pos=dict(
                body=body_pd_joint_delta_pos, balance_passive_force=True
            ),
            pd_joint_pos=dict(body=body_pd_joint_pos, balance_passive_force=True),
        )

    @property
    def _sensor_configs(self):
        return []

    def _after_init(self):
        self.right_hand_finger_link_l_1 = self.robot.links_map["right_two_link"]
        self.right_hand_finger_link_r_1 = self.robot.links_map["right_four_link"]
        self.right_hand_finger_link_r_2 = self.robot.links_map["right_six_link"]
        self.right_tcp = self.robot.links_map["right_tcp_link"]
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
        self.left_tcp = self.robot.links_map["left_tcp_link"]

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

        # disable collisions between fingers. Done in python here instead of the srdf as we can use less collision bits this way and do it more smartly
        # note that the two link of the fingers can collide with other finger links and the palm link so its not included
        link_names = ["one", "three", "four", "five", "six"]
        for ln in link_names:
            self.robot.links_map[f"left_{ln}_link"].set_collision_group_bit(2, 1, 1)
            self.robot.links_map[f"right_{ln}_link"].set_collision_group_bit(2, 2, 1)
        self.robot.links_map["left_palm_link"].set_collision_group_bit(2, 1, 1)
        self.robot.links_map["right_palm_link"].set_collision_group_bit(2, 2, 1)
        self.robot.links_map["left_elbow_roll_link"].set_collision_group_bit(2, 1, 1)
        self.robot.links_map["right_elbow_roll_link"].set_collision_group_bit(2, 2, 1)

        # disable collisions between torso and some other links
        self.robot.links_map["torso_link"].set_collision_group_bit(2, 3, 1)
        self.robot.links_map["left_shoulder_roll_link"].set_collision_group_bit(2, 3, 1)
        self.robot.links_map["right_shoulder_roll_link"].set_collision_group_bit(
            2, 3, 1
        )

    def right_hand_dist_to_open_grasp(self):
        """compute the distance from the current qpos to a open grasp qpos for the right hand"""
        return torch.mean(
            torch.abs(self.robot.qpos[:, self.right_finger_joint_indexes]), dim=1
        )

    def left_hand_dist_to_open_grasp(self):
        """compute the distance from the current qpos to a open grasp qpos for the left hand"""
        return torch.mean(
            torch.abs(self.robot.qpos[:, self.left_finger_joint_indexes]), dim=1
        )

    def left_hand_is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the robot is grasping an object with just its left hand

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.left_hand_finger_link_l_1, object
        )
        r_contact_forces_1 = self.scene.get_pairwise_contact_forces(
            self.left_hand_finger_link_r_1, object
        )
        r_contact_forces_2 = self.scene.get_pairwise_contact_forces(
            self.left_hand_finger_link_r_2, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce_1 = torch.linalg.norm(r_contact_forces_1, axis=1)
        rforce_2 = torch.linalg.norm(r_contact_forces_2, axis=1)

        # direction to open the gripper
        ldirection = self.left_hand_finger_link_l_1.pose.to_transformation_matrix()[
            ..., :3, 1
        ]
        rdirection1 = -self.left_hand_finger_link_r_1.pose.to_transformation_matrix()[
            ..., :3, 1
        ]
        rdirection2 = -self.left_hand_finger_link_r_2.pose.to_transformation_matrix()[
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

    def right_hand_is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the robot is grasping an object with just its right hand

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


@register_agent()
class UnitreeG1UpperBodyWithHeadCamera(UnitreeG1UpperBody):
    uid = "unitree_g1_simplified_upper_body_with_head_camera"

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                "head_camera",
                pose=sapien.Pose(p=[0.05, 0, 0.46], q=euler2quat(0, np.pi / 6, 0)),
                width=128,
                height=128,
                near=0.01,
                far=100,
                fov=np.pi / 2,
                mount=self.robot.links_map["head_link"],
            )
        ]


@register_agent()
class UnitreeG1UpperBodyRightArm(UnitreeG1UpperBody):
    uid = "unitree_g1_simplified_upper_body_right_arm"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/g1_humanoid/g1_simplified_upper_body.urdf"

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
