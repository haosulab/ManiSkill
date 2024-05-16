import numpy as np
import sapien

from mani_skill import ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent


@register_agent()
class WidowX250S(BaseAgent):
    uid = "widowx250s"
    urdf_path = f"{ASSET_DIR}/robots/widowx/wx250s.urdf"
    urdf_config = dict()

    arm_joint_names = [
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
    ]
    gripper_joint_names = ["left_finger", "right_finger"]
    ee_link_name = "ee_gripper_link"

    # System ID results from https://simpler-env.github.io/
    arm_stiffness = [
        1169.7891719504198,
        730.0,
        808.4601346394447,
        1229.1299089624076,
        1272.2760456418862,
        1056.3326605132252,
    ]
    arm_damping = [
        330.0,
        180.0,
        152.12036565582588,
        309.6215302722146,
        201.04998711007383,
        269.51458932695414,
    ]

    # not tuned
    arm_force_limit = [200, 200, 100, 100, 100, 100]
    arm_friction = 0.0
    gripper_stiffness = 1000
    gripper_damping = 200
    gripper_pid_stiffness = 1000
    gripper_pid_damping = 200
    gripper_pid_integral = 300
    gripper_force_limit = 60
    gripper_vel_limit = 0.12
    gripper_acc_limit = 0.50
    gripper_jerk_limit = 5.0

    @property
    def _controller_configs(self):
        _C = {}
        arm_common_args = [
            self.arm_joint_names,
            -1.0,  # dummy limit, which is unused since normalize_action=False
            1.0,
            np.pi / 2,
            1e3,
            1e2,
            # self.arm_stiffness,
            # self.arm_damping,
            # self.arm_force_limit,
        ]
        arm_common_kwargs = dict(
            use_delta=True,
            friction=self.arm_friction,
            ee_link=self.ee_link_name,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            use_delta=True,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_bound=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            normalize_action=False,
        )
        arm_pd_ee_target_delta_pose_align2 = PDEEPoseControllerConfig(
            self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_bound=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            normalize_action=False,
            frame="ee_align",
            use_target=True,
        )

        extra_gripper_clearance = 0.001  # since real gripper is PID, we use extra clearance to mitigate PD small errors; also a trick to have force when grasping
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            0.015 - extra_gripper_clearance,
            0.037 + extra_gripper_clearance,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            normalize_action=True,
            drive_mode="force",
        )
        _C["arm"] = dict(
            arm_pd_ee_delta_pose=arm_pd_ee_delta_pose,
            arm_pd_ee_target_delta_pose_align2=arm_pd_ee_target_delta_pose_align2,
            arm_pd_joint_delta_pos=arm_pd_joint_delta_pos,
        )

        _C["gripper"] = dict(
            gripper_pd_joint_pos=gripper_pd_joint_pos,
        )
        controller_configs = {}
        for arm_controller_name in _C["arm"]:
            for gripper_controller_name in _C["gripper"]:
                c = {}
                c["arm"] = _C["arm"][arm_controller_name]
                c["gripper"] = _C["gripper"][gripper_controller_name]
                combined_name = arm_controller_name + "_" + gripper_controller_name
                controller_configs[combined_name] = c

        return deepcopy_dict(controller_configs)
