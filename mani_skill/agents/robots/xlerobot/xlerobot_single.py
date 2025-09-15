from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.types import Array

XLEROBOT_SINGLE_WHEELS_COLLISION_BIT = 30
"""Collision bit of the xlerobot robot wheel links"""
XLEROBOT_SINGLE_BASE_COLLISION_BIT = 31
"""Collision bit of the xlerobot base"""


@register_agent()
class XlerobotSingle(BaseAgent):
    uid = "xlerobot_single"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/xlerobot/xlerobot_single.urdf"
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            Fixed_Jaw=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            Moving_Jaw=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            pose=sapien.Pose(),
            qpos=np.array([0, 0, 0, 0, 0, 0.0, 0.303, 0.303, 0, 0, 0]),  # Updated DOFs for Fetch with single SO100 arm (11 joints)
        )
    )

    @property
    def _sensor_configs(self):
        """
        Configure cameras for Fetch robot with single-arm setup
        
        Camera configuration includes:
        - head_camera: Main head camera for workspace overview
        - right_arm_camera: Hand-mounted camera on the arm for precise manipulation
        """
        return [
            # HEAD CAMERA - Main workspace overview camera
            CameraConfig(
                uid="fetch_head",
                pose=Pose.create_from_pq([0, 0, 0], [1, 0, 0, 0]),  # Identity transform
                width=256,
                height=256,
                fov=1.6,  # Wide field of view for workspace monitoring
                near=0.01,
                far=100,
                entity_uid="head_camera_link",  # Mount to dedicated head camera link
            ),
            
            # ARM CAMERA - Hand-mounted camera for precise manipulation
            CameraConfig(
                uid="fetch_right_arm_camera", 
                pose=Pose.create_from_pq([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),  # Identity transform
                width=128,
                height=128,
                fov=1.3,  # Wide field of view for workspace monitoring
                near=0.01,
                far=100,
                entity_uid="Right_Arm_Camera",  # Mount to dedicated camera link
            ),
        ]

    def __init__(self, *args, **kwargs):
        # Arm
        self.arm_joint_names = [
            "Rotation",
            "Pitch",
            "Elbow",
            "Wrist_Pitch",
            "Wrist_Roll",
        ]
        self.arm_stiffness = 2e4
        self.arm_damping = 1e2
        self.arm_force_limit = 250

        self.gripper_joint_names = [
            "Jaw",
        ]
        self.gripper_stiffness = 50
        self.gripper_damping = 1e2
        self.gripper_force_limit = 2.8

        self.ee_link_name = "Fixed_Jaw"

        self.body_joint_names = [
            "head_pan_joint",
            "head_tilt_joint",
        ]
        self.body_stiffness = 1e4
        self.body_damping = 1e2
        self.body_force_limit = 200

        self.base_joint_names = [
            "root_x_axis_joint",
            "root_y_axis_joint",
            "root_z_rotation_joint",
        ]

        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )

        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # PD ee position (for human-interaction/teleoperation)
        arm_pd_ee_delta_pose_align = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_delta_pose_align.frame = "ee_align"

        # PD joint velocity
        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,  # this might need to be tuned separately
            self.arm_force_limit,
        )

        # PD joint position and velocity
        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=True,
        )
        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        # For SO100 gripper, we use a regular PDJointPosController instead of mimic controller
        gripper_pd_joint_pos = PDJointPosControllerConfig(
            self.gripper_joint_names,
            -20,  # closed position - update this value if needed
            20,  # open position - update this value if needed
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
        )

        # -------------------------------------------------------------------------- #
        # Body
        # -------------------------------------------------------------------------- #
        body_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.body_joint_names,
            -0.1,
            0.1,
            self.body_stiffness,
            self.body_damping,
            self.body_force_limit,
            use_delta=True,
        )

        # useful to keep body unmoving from passed position
        stiff_body_pd_joint_pos = PDJointPosControllerConfig(
            self.body_joint_names,
            None,
            None,
            1e5,
            1e5,
            1e5,
            normalize_action=False,
        )

        # -------------------------------------------------------------------------- #
        # Base
        # -------------------------------------------------------------------------- #
        base_pd_joint_vel = PDBaseForwardVelControllerConfig(
            self.base_joint_names,
            lower=[-1, -3.14],
            upper=[1, 3.14],
            damping=1000,
            force_limit=500,
        )


        
        # Create a single-arm controller config
        controller_configs = dict(
            pd_joint_delta_pos=dict(
                base=base_pd_joint_vel,
                arm1=arm_pd_joint_delta_pos,
                gripper1=gripper_pd_joint_pos,
                body=body_pd_joint_delta_pos,
            ),
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
            pd_ee_delta_pos=dict(
                arm=arm_pd_ee_delta_pos,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
            pd_ee_delta_pose_align=dict(
                arm=arm_pd_ee_delta_pose_align,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
            # TODO(jigu): how to add boundaries for the following controllers
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
            pd_ee_target_delta_pos=dict(
                arm=arm_pd_ee_target_delta_pos,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
            # Caution to use the following controllers
            pd_joint_vel=dict(
                arm=arm_pd_joint_vel,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
            pd_joint_pos_vel=dict(
                arm=arm_pd_joint_pos_vel,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
            pd_joint_delta_pos_vel=dict(
                arm=arm_pd_joint_delta_pos_vel,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
            pd_joint_delta_pos_stiff_body=dict(
                arm=arm_pd_joint_delta_pos,
                gripper=gripper_pd_joint_pos,
                body=stiff_body_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy(controller_configs)

    def _after_init(self):
        # Arm
        self.finger1_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "Fixed_Jaw"
        )
        self.finger2_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "Moving_Jaw"
        )
        self.finger1_tip: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "Fixed_Jaw_tip"
        )
        self.finger2_tip: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "Moving_Jaw_tip"
        )
        self.tcp: Link = self.finger1_link

        self.base_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "base_link"
        )
        self.top_base_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "top_base_link"
        )
        for link in [self.top_base_link]:
            link.set_collision_group_bit(
                group=2, bit_idx=XLEROBOT_SINGLE_WHEELS_COLLISION_BIT, bit=1
            )
        self.base_link.set_collision_group_bit(
            group=2, bit_idx=XLEROBOT_SINGLE_BASE_COLLISION_BIT, bit=1
        )

        self.head_camera_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "head_camera_link"
        )

        self.queries: Dict[
            str, Tuple[physx.PhysxGpuContactPairImpulseQuery, Tuple[int]]
        ] = dict()

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=110):
        """Check if the robot is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # direction to open the gripper
        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold=0.2):
        qvel = self.robot.get_qvel()[
            :, 3:-1
        ]  # exclude the base joints
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """Build a grasp pose (panda_hand_tcp)."""
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)

    @property
    def tcp_pos(self):
        # computes the tool center point as the mid point between the the fixed and moving jaw's tips
        return (self.finger1_tip.pose.p + self.finger2_tip.pose.p) / 2

    @property
    def tcp_pose(self):
        return Pose.create_from_pq(self.tcp_pos, self.finger1_link.pose.q)


