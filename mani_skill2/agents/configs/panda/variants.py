from .defaults import PandaDefaultConfig
from mani_skill2.agents.controllers import *
from mani_skill2.agents.camera import MountedCameraConfig
import numpy as np


class PandaPourConfig(PandaDefaultConfig):
    @property
    def controllers(self):
        controllers = super().controllers

        arm_pd_ee_pose = PDEEPoseControllerConfig(
            self.arm_joint_names,
            -100,
            100,
            np.pi,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            ee_link=self.ee_link_name,
            use_delta=False,
            frame="base",
            normalize_action=False,
        )

        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            -0.01,
            0.04,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
        )

        controllers["pd_ee_pose"] = dict(
            arm=arm_pd_ee_pose, gripper=gripper_pd_joint_pos
        )

        return controllers


class PandaBucketConfig(PandaDefaultConfig):
    def __init__(self) -> None:
        super().__init__()

        self.urdf_path = "{description}/panda_bucket.urdf"
        self.ee_link_name = "bucket"

    @property
    def controllers(self):
        controller_configs = super().controllers
        for k, v in controller_configs.items():
            if isinstance(v, dict) and "gripper" in v:
                v.pop("gripper")
        return controller_configs

    @property
    def cameras(self):
        return dict(
            hand_camera=MountedCameraConfig(
                mount_link="bucket",
                mount_p=[0.0, 0.08, 0.0],
                mount_q=[0.5, -0.5, -0.5, -0.5],
                hide_mount_link=False,
                width=128,
                height=128,
                near=0.01,
                far=10,
                fx=64,
                fy=64,
            )
        )


class PandaPinConfig(PandaDefaultConfig):
    def __init__(self) -> None:
        super().__init__()

        self.urdf_path = "{description}/panda_rolling_pin.urdf"
        self.ee_link_name = "rolling_pin"

    @property
    def controllers(self):
        controller_configs = super().controllers
        for k, v in controller_configs.items():
            if isinstance(v, dict) and "gripper" in v:
                v.pop("gripper")
        return controller_configs

    @property
    def cameras(self):
        return dict(
            hand_camera=MountedCameraConfig(
                mount_link="rolling_pin",
                mount_p=[0.0, 0.0, 0.0],
                mount_q=[0.0, 0, 0.0, 0],
                hide_mount_link=False,
                width=128,
                height=128,
                near=0.01,
                far=10,
                fx=64,
                fy=64,
            )
        )


class PandaStickConfig(PandaDefaultConfig):
    def __init__(self) -> None:
        super().__init__()

        self.urdf_path = "{description}/panda_stick.urdf"
        self.ee_link_name = "panda_hand"

    @property
    def controllers(self):
        controller_configs = super().controllers

        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            ee_link=self.ee_link_name,
            frame="base",
            normalize_action=False,
        )

        controller_configs["pd_ee_delta_pose_demo"] = dict(arm=arm_pd_ee_delta_pose)

        for k, v in controller_configs.items():
            if isinstance(v, dict) and "gripper" in v:
                v.pop("gripper")
        return controller_configs
