from mani_skill2.agents.camera import MountedCameraConfig
from mani_skill2.agents.controllers import *


class Xmate3RobotiqDefaultConfig:
    def __init__(self) -> None:
        self.urdf_path = "{description}/fixed_xmate3_robotiq.urdf"
        self.urdf_config = dict(
            _materials=dict(
                gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
            ),
            link=dict(
                panda_leftfinger=dict(
                    material="gripper", patch_radius=0.1, min_patch_radius=0.1
                ),
                panda_rightfinger=dict(
                    material="gripper", patch_radius=0.1, min_patch_radius=0.1
                ),
            ),
        )

        self.arm_joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 100

        self.gripper_joint_names = [
            "robotiq_2f_140_left_driver_joint",
            "robotiq_2f_140_right_driver_joint",
        ]
        self.gripper_stiffness = 1e3
        self.gripper_damping = 1e2
        self.gripper_force_limit = 100

        self.ee_link_name = "grasp_convenient_link"

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
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            -0.05,
            0.05,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )

        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            self.arm_joint_names,
            -0.01,
            0.01,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            ee_link=self.ee_link_name,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            self.arm_joint_names,
            -0.01,
            0.01,
            -0.05,
            0.05,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            ee_link=self.ee_link_name,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        # NOTE(jigu): IssacGym uses large P and D but with force limit
        # However, tune a good force limit to have a good mimic behavior
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            0,
            0.068 + 0.01,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
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
        return dict(
            base_camera=MountedCameraConfig(
                mount_link="camera_base_link",
                mount_p=[0.0, 0.0, 0.0],
                mount_q=[1, 0, 0, 0],
                hide_mount_link=True,
                width=128,
                height=128,
                near=0.01,
                far=10,
                fx=64,
                fy=64,
            ),
            hand_camera=MountedCameraConfig(
                mount_link="camera_hand_link",
                mount_p=[0.0, 0.0, 0.0],
                mount_q=[1, 0, 0, 0],
                hide_mount_link=True,
                width=128,
                height=128,
                near=0.01,
                far=10,
                fx=64,
                fy=64,
            ),
        )
