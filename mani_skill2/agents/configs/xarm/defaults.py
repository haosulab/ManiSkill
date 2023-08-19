from copy import deepcopy
import numpy as np

from mani_skill2.agents.controllers import *
from mani_skill2.sensors.camera import CameraConfig


class XArmDefaultConfig:
    def __init__(self, sim_params) -> None:
        self.urdf_config = dict(
            _materials=dict(
                gripper=dict(static_friction=sim_params['robot_fri_static'], 
                            dynamic_friction=sim_params['robot_fri_dynamic'], 
                            restitution=sim_params['robot_restitution'])
            ),
            link=dict(
                left_finger=dict(
                    material="gripper", patch_radius=0.1, min_patch_radius=0.1
                ),
                right_finger=dict(
                    material="gripper", patch_radius=0.1, min_patch_radius=0.1
                ),
            ),
        )

        self.arm_stiffness = sim_params['stiffness']    # NOTE: increase from 1e3
        self.arm_damping = sim_params['damping']  # NOTE: increase from 1e2
        self.arm_force_limit = sim_params['force_limit']

        self.gripper_joint_names = [
            "left_finger_joint",
            "right_finger_joint",
        ]
        self.gripper_stiffness = sim_params['stiffness']    # NOTE: increase from 1e3
        self.gripper_damping = sim_params['damping']  # NOTE: increase from 1e2
        self.gripper_force_limit = sim_params['force_limit']

        self.ee_link_name = "link_tcp"

        self.arm_use_target = True # NOTE(chichu): enable to calculate next_target_pose based on current_target_pose, making it easier for sim2real

    @property
    def controllers(self):
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
            use_target=self.arm_use_target, 
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
            use_target=self.arm_use_target,
        )

        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            ee_link=self.ee_link_name,
            use_target=self.arm_use_target,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            ee_link=self.ee_link_name,
            use_target=self.arm_use_target,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        # NOTE(jigu): IssacGym uses large P and D but with force limit
        # However, tune a good force limit to have a good mimic behavior
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            -0.01,  # a trick to have force when the object is thin
            0.0446430,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            use_target=self.arm_use_target,
        )

        controller_configs = dict(
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos),
            arm_pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose, gripper=gripper_pd_joint_pos
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    @property
    def cameras(self):
        return CameraConfig(
            uid="hand_camera",
            p=[-0.0464982, 0.0200011, 0.0360011],
            q=[-0.70710678, 0, 0.70710678, 0],
            width=128,
            height=128,
            fov=1.57,
            near=0.01,
            far=10,
            actor_uid="xarm_gripper_base_link",
        )


class XArm7DefaultConfig(XArmDefaultConfig):
    def __init__(self, sim_params) -> None:
        super().__init__(sim_params)
        self.urdf_path = "{PACKAGE_ASSET_DIR}/descriptions/xarm7_textured_with_gripper_reduced_dof_v2.urdf"
        self.arm_joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]


class XArm7D435DefaultConfig(XArmDefaultConfig):
    def __init__(self) -> None:
        super().__init__()
        self.urdf_path = "{PACKAGE_ASSET_DIR}/descriptions/xarm7_textured_with_gripper_reduced_dof_d435_v2.urdf"
        self.arm_joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]

    @property
    def cameras(self):
        return CameraConfig(
            uid="hand_camera",
            p=[0, 0, 0],
            q=[1, 0, 0, 0],
            width=848,
            height=480,
            fov=np.deg2rad(43.5),
            near=0.01,
            far=10,
            actor_uid="camera_color_frame",
        )
