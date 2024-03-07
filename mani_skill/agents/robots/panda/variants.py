## TODO (stao): These variants are maintained for some ManiSkill2/1 tasks. We should remove when we can
import numpy as np

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.controllers import *
from mani_skill.sensors.camera import CameraConfig

from .panda import Panda


class PandaPour(Panda):
    @property
    def controller_configs(self):
        controller_configs = super().controller_configs

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

        controller_configs["pd_ee_pose"] = dict(
            arm=arm_pd_ee_pose, gripper=gripper_pd_joint_pos
        )

        return controller_configs


class PandaBucket(Panda):
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_bucket.urdf"

    def __init__(self) -> None:
        super().__init__()
        self.ee_link_name = "bucket"

    @property
    def controller_configs(self):
        controller_configs = super().controller_configs
        for k, v in controller_configs.items():
            if isinstance(v, dict) and "gripper" in v:
                v.pop("gripper")
        return controller_configs

    sensor_configs = [
        CameraConfig(
            uid="hand_camera",
            p=[0.0, 0.08, 0.0],
            q=[0.5, -0.5, -0.5, -0.5],
            width=128,
            height=128,
            near=0.01,
            far=100,
            fov=np.pi / 2,
            entity_uid="bucket",
        )
    ]


class PandaStick(Panda):
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_stick.urdf"

    def __init__(self) -> None:
        super().__init__()
        self.ee_link_name = "panda_hand"

    @property
    def controller_configs(self):
        controller_configs = super().controller_configs

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


class PandaPinch(Panda):
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_pinch.urdf"

    def __init__(self) -> None:
        super().__init__()

    @property
    def controller_configs(self):
        controller_configs = super().controller_configs

        for key in controller_configs:
            controller_configs[key]["gripper"].upper = 0.06

        return controller_configs
