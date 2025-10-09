from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.types import Array

FETCH_WHEELS_COLLISION_BIT = 30
"""Collision bit of the fetch robot wheel links"""
FETCH_BASE_COLLISION_BIT = 31
"""Collision bit of the fetch base"""


@register_agent(asset_download_ids=["xlerobot"])
class Xlerobot(BaseAgent):
    uid = "xlerobot"
    urdf_path = f"{ASSET_DIR}/robots/xlerobot/xlerobot.urdf"
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
            Fixed_Jaw_2=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            Moving_Jaw_2=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            pose=sapien.Pose(),
            # qpos with the actual mapping used in gen_spawn_positions_xlerobot.py:
            # Index:  0  1  2  3   4   5    6     7     8    9     10    11   12   13    14   15  16
            # Joint: [x, y, r, arm1, arm2, head, arm1, arm2, head, arm1, arm2, arm1, arm2, arm1, arm2, g1, g2]
            # Where arm indices are:
            # - First arm: [3,6,9,11,13] (5 joints: Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll)
            # - Second arm: [4,7,10,12,14] (5 joints: Rotation_2, Pitch_2, Elbow_2, Wrist_Pitch_2, Wrist_Roll_2)
            # - Base: [0,1,2] (x, y, rotation)
            # - Head: [5,8] (pan, tilt)
            # - Grippers: [15,16] (Jaw, Jaw_2)
            qpos=np.array([0, 0, 0,           # [0,1,2] base: x, y, rotation
                          0, 0,              # [3,4] first values for arm1, arm2
                          0,                 # [5] head pan
                          3.14, 3.14,      # [6,7] second values for arm1, arm2  
                          0,                 # [8] head tilt
                          3.14, 3.14,      # [9,10] third values for arm1, arm2
                          0, 0,              # [11,12] fourth values for arm1, arm2
                          1.57, 1.57,          # [13,14] fifth values for arm1, arm2
                          0, 0]),            # [15,16] grippers: Jaw, Jaw_2
        )
    )

    @property
    def _sensor_configs(self):
        """
        Configure cameras for Fetch robot with dual-arm setup
        
        Camera configuration includes:
        - head_camera: Main head camera for workspace overview
        - right_arm_camera: Hand-mounted camera on first arm for precise manipulation
        - left_arm_camera: Hand-mounted camera on second arm for precise manipulation
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
            
            # FIRST ARM CAMERA - Hand-mounted camera for precise manipulation
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
            
            # SECOND ARM CAMERA - Hand-mounted camera for precise manipulation
            CameraConfig(
                uid="fetch_left_arm_camera",
                pose=Pose.create_from_pq([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),  # Identity transform
                width=128,
                height=128,
                fov=1.3,  # Wide field of view for workspace monitoring
                near=0.01,
                far=100,
                entity_uid="Left_Arm_Camera",  # Mount to dedicated camera link
            ),
        ]

    def __init__(self, *args, **kwargs):
        # First arm
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
        
        # Second arm
        self.arm2_joint_names = [
            "Rotation_2",
            "Pitch_2",
            "Elbow_2",
            "Wrist_Pitch_2",
            "Wrist_Roll_2",
        ]
        self.arm2_stiffness = 2e4
        self.arm2_damping = 1e2
        self.arm2_force_limit = 250

        self.gripper2_joint_names = [
            "Jaw_2",
        ]
        self.gripper2_stiffness = 50
        self.gripper2_damping = 1e2
        self.gripper2_force_limit = 2.8

        self.ee2_link_name = "Fixed_Jaw_2"

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

        # Add controller configs for second arm
        arm2_pd_joint_pos = PDJointPosControllerConfig(
            self.arm2_joint_names,
            None,
            None,
            self.arm2_stiffness,
            self.arm2_damping,
            self.arm2_force_limit,
            normalize_action=False,
        )
        arm2_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm2_joint_names,
            -0.1,
            0.1,
            self.arm2_stiffness,
            self.arm2_damping,
            self.arm2_force_limit,
            use_delta=True,
        )
        
        # For SO100 gripper, we use a regular PDJointPosController instead of mimic controller
        gripper2_pd_joint_pos = PDJointPosControllerConfig(
            self.gripper2_joint_names,
            -20,  # closed position - update this value if needed
            20,  # open position - update this value if needed
            self.gripper2_stiffness,
            self.gripper2_damping,
            self.gripper2_force_limit,
        )
        
        # Create a dual-arm controller config
        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos,
                gripper=gripper_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
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
            # Default single-arm controller configs
            pd_joint_delta_pos_arm2=dict(
                arm=arm2_pd_joint_delta_pos,
                gripper=gripper2_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
            pd_joint_pos_arm2=dict(
                arm=arm2_pd_joint_pos,
                gripper=gripper2_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
            pd_joint_target_delta_pos_arm2=dict(
                arm=arm_pd_joint_target_delta_pos,
                gripper=gripper2_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
            pd_joint_vel_arm2=dict(
                arm=arm_pd_joint_vel,
                gripper=gripper2_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
            pd_joint_pos_vel_arm2=dict(
                arm=arm_pd_joint_pos_vel,
                gripper=gripper2_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
            pd_joint_delta_pos_vel_arm2=dict(
                arm=arm_pd_joint_delta_pos_vel,
                gripper=gripper2_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
            pd_joint_delta_pos_stiff_body_arm2=dict(
                arm=arm2_pd_joint_delta_pos,
                gripper=gripper2_pd_joint_pos,
                body=stiff_body_pd_joint_pos,
                base=base_pd_joint_vel,
            ),
            # Default dual-arm controller configs
            pd_joint_delta_pos_dual_arm=dict(
                base=base_pd_joint_vel,
                arm1=arm_pd_joint_delta_pos,
                arm2=arm2_pd_joint_delta_pos,
                gripper1=gripper_pd_joint_pos,
                gripper2=gripper2_pd_joint_pos,
                body=body_pd_joint_delta_pos,
            ),
            pd_joint_pos_dual_arm=dict(
                arm1=arm_pd_joint_pos,
                arm2=arm2_pd_joint_pos,
                gripper1=gripper_pd_joint_pos,
                gripper2=gripper2_pd_joint_pos,
                body=body_pd_joint_delta_pos,
                base=base_pd_joint_vel,
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy(controller_configs)

    def _after_init(self):
        # First arm
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
        
        # Second arm
        self.finger1_link_2: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "Fixed_Jaw_2"
        )
        self.finger2_link_2: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "Moving_Jaw_2"
        )
        self.finger1_tip_2: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "Fixed_Jaw_2_tip"
        )
        self.finger2_tip_2: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "Moving_Jaw_2_tip"
        )
        self.tcp_2: Link = self.finger1_link_2

        self.base_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "base_link"
        )
        self.top_base_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "top_base_link"
        )
        for link in [self.top_base_link]:
            link.set_collision_group_bit(
                group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1
            )
        self.base_link.set_collision_group_bit(
            group=2, bit_idx=FETCH_BASE_COLLISION_BIT, bit=1
        )

        self.head_camera_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "head_camera_link"
        )

        self.queries: Dict[
            str, Tuple[physx.PhysxGpuContactPairImpulseQuery, Tuple[int]]
        ] = dict()

    def get_arm_joint_indices(self):
        """
        Get the joint indices for each arm according to the current qpos mapping.
        
        Returns:
            dict: Dictionary with keys 'arm1', 'arm2', 'base', 'head', 'grippers'
        """
        return {
            'base': [0, 1, 2],              # x, y, rotation
            'arm1': [3, 6, 9, 11, 13],      # Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll
            'arm2': [4, 7, 10, 12, 14],     # Rotation_2, Pitch_2, Elbow_2, Wrist_Pitch_2, Wrist_Roll_2
            'head': [5, 8],                 # pan, tilt
            'grippers': [15, 16]            # Jaw, Jaw_2
        }

    def map_full_joints_to_current(self, full_joints):
        """
        Map from full_joints (17 elements) to current joint ordering.
        
        Mapping according to specification:
        - full_joints[0,2] → current_joints[0,1] (base x position and base rotation)
        - full_joints[3,6,9,11,13] → current_joints[2,3,4,5,6] (first arm joints)
        - full_joints[4,7,10,12,14] → current_joints[7,8,9,10,11] (second arm joints)
        - full_joints[15] → current_joints[15] (first arm gripper)
        - full_joints[16] → current_joints[16] (second arm gripper)
        - full_joints[1] → current_joints[1] (base y - assuming continuous)
        - full_joints[5,8] → current_joints[3,4] (head joints)
        
        Args:
            full_joints: Array of shape (..., 17) with the original joint ordering
            
        Returns:
            current_joints: Array of shape (..., 17) with the current joint ordering
        """
        if torch.is_tensor(full_joints):
            current_joints = torch.zeros_like(full_joints)
        else:
            current_joints = np.zeros_like(full_joints)
            
        # Base joints: x, y, rotation
        current_joints[..., 0] = full_joints[..., 0]  # base x
        current_joints[..., 1] = full_joints[..., 1]  # base y  
        current_joints[..., 2] = full_joints[..., 2]  # base rotation
        
        # Head joints
        current_joints[..., 3] = full_joints[..., 5]  # head pan
        current_joints[..., 4] = full_joints[..., 8]  # head tilt
        
        # First arm joints: [3,6,9,11,13] → [5,6,7,8,9]
        current_joints[..., 5] = full_joints[..., 3]   # Rotation
        current_joints[..., 6] = full_joints[..., 6]   # Pitch
        current_joints[..., 7] = full_joints[..., 9]   # Elbow
        current_joints[..., 8] = full_joints[..., 11]  # Wrist_Pitch
        current_joints[..., 9] = full_joints[..., 13]  # Wrist_Roll
        
        # Second arm joints: [4,7,10,12,14] → [10,11,12,13,14]
        current_joints[..., 10] = full_joints[..., 4]   # Rotation_2
        current_joints[..., 11] = full_joints[..., 7]   # Pitch_2
        current_joints[..., 12] = full_joints[..., 10]  # Elbow_2
        current_joints[..., 13] = full_joints[..., 12]  # Wrist_Pitch_2
        current_joints[..., 14] = full_joints[..., 14]  # Wrist_Roll_2
        
        # Gripper joints
        current_joints[..., 15] = full_joints[..., 15]  # Jaw
        current_joints[..., 16] = full_joints[..., 16]  # Jaw_2
        
        return current_joints

    def map_current_joints_to_full(self, current_joints):
        """
        Map from current joint ordering back to full_joints ordering.
        This is the inverse of map_full_joints_to_current.
        
        Args:
            current_joints: Array of shape (..., 17) with the current joint ordering
            
        Returns:
            full_joints: Array of shape (..., 17) with the original joint ordering
        """
        if torch.is_tensor(current_joints):
            full_joints = torch.zeros_like(current_joints)
        else:
            full_joints = np.zeros_like(current_joints)
            
        # Base joints
        full_joints[..., 0] = current_joints[..., 0]   # base x
        full_joints[..., 1] = current_joints[..., 1]   # base y
        full_joints[..., 2] = current_joints[..., 2]   # base rotation
        
        # First arm joints: [5,6,7,8,9] → [3,6,9,11,13]
        full_joints[..., 3] = current_joints[..., 5]   # Rotation
        full_joints[..., 6] = current_joints[..., 6]   # Pitch
        full_joints[..., 9] = current_joints[..., 7]   # Elbow
        full_joints[..., 11] = current_joints[..., 8]  # Wrist_Pitch
        full_joints[..., 13] = current_joints[..., 9]  # Wrist_Roll
        
        # Second arm joints: [10,11,12,13,14] → [4,7,10,12,14]
        full_joints[..., 4] = current_joints[..., 10]  # Rotation_2
        full_joints[..., 7] = current_joints[..., 11]  # Pitch_2
        full_joints[..., 10] = current_joints[..., 12] # Elbow_2
        full_joints[..., 12] = current_joints[..., 13] # Wrist_Pitch_2
        full_joints[..., 14] = current_joints[..., 14] # Wrist_Roll_2
        
        # Head joints
        full_joints[..., 5] = current_joints[..., 3]   # head pan
        full_joints[..., 8] = current_joints[..., 4]   # head tilt
        
        # Gripper joints
        full_joints[..., 15] = current_joints[..., 15] # Jaw
        full_joints[..., 16] = current_joints[..., 16] # Jaw_2
        
        return full_joints

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=110, arm_id=None):
        """Check if the robot is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 110.
            arm_id (int, optional): Which arm to check (1 for first arm, 2 for second arm). 
                                   If None (default), check both arms and return True if either is grasping.
        """
        if arm_id is None:
            # Check both arms, return True if either is grasping
            arm1_grasping = self._check_single_arm_grasping(object, min_force, max_angle, arm_id=1)
            arm2_grasping = self._check_single_arm_grasping(object, min_force, max_angle, arm_id=2)
            return torch.logical_or(arm1_grasping, arm2_grasping)
        else:
            # Check specific arm
            return self._check_single_arm_grasping(object, min_force, max_angle, arm_id)

    def _check_single_arm_grasping(self, object: Actor, min_force=0.5, max_angle=110, arm_id=1):
        """Internal method to check grasping for a specific arm"""
        if arm_id == 1:
            finger1_link = self.finger1_link
            finger2_link = self.finger2_link
        elif arm_id == 2:
            finger1_link = self.finger1_link_2
            finger2_link = self.finger2_link_2
        else:
            raise ValueError(f"Invalid arm_id: {arm_id}. Must be 1 or 2.")
            
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # direction to open the gripper
        ldirection = finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold=0.2, base_threshold: float = 0.05):
        qvel = self.robot.get_qvel()[
            :, 3:-2
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

    @property
    def tcp_pos_2(self):
        # computes the tool center point as the mid point between the the fixed and moving jaw's tips
        return (self.finger1_tip_2.pose.p + self.finger2_tip_2.pose.p) / 2

    @property
    def tcp_pose_2(self):
        return Pose.create_from_pq(self.tcp_pos_2, self.finger1_link_2.pose.q)
