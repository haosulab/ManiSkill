from copy import deepcopy

import numpy as np

from mani_skill2.agents.controllers import *
from mani_skill2.sensors.camera import CameraConfig


class MobilePandaDualArmDefaultConfig:
    def __init__(self) -> None:
        self.urdf_path = "{PACKAGE_ASSET_DIR}/descriptions/mobile_panda_dual_arm.urdf"
        self.urdf_config = dict(
            _materials=dict(
                gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
            ),
            link=dict(
                right_panda_leftfinger=dict(
                    material="gripper", patch_radius=0.1, min_patch_radius=0.1
                ),
                right_panda_rightfinger=dict(
                    material="gripper", patch_radius=0.1, min_patch_radius=0.1
                ),
                left_panda_leftfinger=dict(
                    material="gripper", patch_radius=0.1, min_patch_radius=0.1
                ),
                left_panda_rightfinger=dict(
                    material="gripper", patch_radius=0.1, min_patch_radius=0.1
                ),
            ),
        )

        self.base_joint_names = [
            "root_x_axis_joint",
            "root_y_axis_joint",
            "root_z_rotation_joint",
            "linear_actuator_height",
        ]

        arm_joint_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]
        self.arm_joint_names = {
            "left": ["left_" + x for x in arm_joint_names],
            "right": ["right_" + x for x in arm_joint_names],
        }
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 100
        self.arm_joint_delta = 0.1
        self.arm_ee_delta = 0.1

        gripper_joint_names = [
            "panda_finger_joint1",
            "panda_finger_joint2",
        ]
        self.gripper_joint_names = {
            "left": ["left_" + x for x in gripper_joint_names],
            "right": ["right_" + x for x in gripper_joint_names],
        }
        self.gripper_stiffness = 1e3
        self.gripper_damping = 1e2
        self.gripper_force_limit = 100

        self.ee_link_name = {
            "left": "left_panda_hand_tcp",
            "right": "right_panda_hand_tcp",
        }

        self.camera_h = 1.5

    @property
    def controllers(self):
        _C = {}  # controller configs for each component

        # -------------------------------------------------------------------------- #
        # Mobile Base
        # -------------------------------------------------------------------------- #
        # fmt: off
        _C["base"] = dict(
            # PD ego-centric joint velocity
            base_pd_joint_vel=PDBaseVelControllerConfig(
                self.base_joint_names,
                lower=[-0.5, -0.5, -3.14, -0.5],
                upper=[0.5, 0.5, 3.14, 0.5],
                damping=1000,
                force_limit=500,
            )
        )
        # fmt: on

        sides = list(self.arm_joint_names.keys())

        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        for side in sides:
            # PD joint velocity
            arm_pd_joint_vel = PDJointVelControllerConfig(
                self.arm_joint_names[side],
                lower=-3.14,
                upper=3.14,
                damping=self.arm_damping,
                force_limit=self.arm_force_limit,
            )

            # PD joint position
            arm_pd_joint_delta_pos = PDJointPosControllerConfig(
                self.arm_joint_names[side],
                -self.arm_joint_delta,
                self.arm_joint_delta,
                stiffness=self.arm_stiffness,
                damping=self.arm_damping,
                force_limit=self.arm_force_limit,
                use_delta=True,
            )
            arm_pd_joint_pos = PDJointPosControllerConfig(
                self.arm_joint_names[side],
                None,
                None,
                stiffness=self.arm_stiffness,
                damping=self.arm_damping,
                force_limit=self.arm_force_limit,
                normalize_action=False,
            )
            arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
            arm_pd_joint_target_delta_pos.use_target = True

            # PD ee pose
            arm_pd_ee_delta_pos = PDEEPosControllerConfig(
                self.arm_joint_names[side],
                -self.arm_ee_delta,
                self.arm_ee_delta,
                stiffness=self.arm_stiffness,
                damping=self.arm_damping,
                force_limit=self.arm_force_limit,
                ee_link=self.ee_link_name[side],
            )
            arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
                self.arm_joint_names[side],
                -self.arm_ee_delta,
                self.arm_ee_delta,
                rot_bound=self.arm_ee_delta,
                stiffness=self.arm_stiffness,
                damping=self.arm_damping,
                force_limit=self.arm_force_limit,
                ee_link=self.ee_link_name[side],
            )

            arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
            arm_pd_ee_target_delta_pos.use_target = True
            arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
            arm_pd_ee_target_delta_pose.use_target = True

            # PD joint position and velocity
            arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
                self.arm_joint_names[side],
                None,
                None,
                stiffness=self.arm_stiffness,
                damping=self.arm_damping,
                force_limit=self.arm_force_limit,
                normalize_action=False,
            )
            arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
                self.arm_joint_names[side],
                -self.arm_joint_delta,
                self.arm_joint_delta,
                stiffness=self.arm_stiffness,
                damping=self.arm_damping,
                force_limit=self.arm_force_limit,
                use_delta=True,
            )

            _C[side + "_arm"] = dict(
                arm_pd_joint_vel=arm_pd_joint_vel,
                arm_pd_joint_delta_pos=arm_pd_joint_delta_pos,
                arm_pd_joint_pos=arm_pd_joint_pos,
                arm_pd_joint_target_delta_pos=arm_pd_joint_target_delta_pos,
                arm_pd_ee_delta_pos=arm_pd_ee_delta_pos,
                arm_pd_ee_delta_pose=arm_pd_ee_delta_pose,
                arm_pd_ee_target_delta_pos=arm_pd_ee_target_delta_pos,
                arm_pd_ee_target_delta_pose=arm_pd_ee_target_delta_pose,
                arm_pd_joint_pos_vel=arm_pd_joint_pos_vel,
                arm_pd_joint_delta_pos_vel=arm_pd_joint_delta_pos_vel,
            )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        for side in sides:
            _C[side + "_gripper"] = dict(
                gripper_pd_joint_pos=PDJointPosMimicControllerConfig(
                    self.gripper_joint_names[side],
                    -0.01,  # a trick to have force when the object is thin
                    0.04,
                    self.gripper_stiffness,
                    self.gripper_damping,
                    self.gripper_force_limit,
                ),
            )

        controller_configs = {}
        for base_controller_name in _C["base"]:
            # Assume right arm always be there
            for arm_controller_name in _C["right_arm"]:
                c = {"base": _C["base"][base_controller_name]}
                for side in sides:
                    c[side + "_arm"] = _C[side + "_arm"][arm_controller_name]
                    # Assume gripper_pd_joint_pos only
                    c[side + "_gripper"] = _C[side + "_gripper"]["gripper_pd_joint_pos"]
                combined_name = base_controller_name + "_" + arm_controller_name
                controller_configs[combined_name] = c

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    @property
    def cameras(self):
        cameras = []
        qs = [
            [0.9238795, 0, 0.3826834, 0],
            [0.46193977, 0.33141357, 0.19134172, -0.80010315],
            [-0.46193977, 0.33141357, -0.19134172, -0.80010315],
        ]
        for i in range(3):
            # q = transforms3d.euler.euler2quat(-np.pi/3*i, np.pi/4, 0, 'rzyx')
            q = qs[i]
            camera = CameraConfig(
                f"overhead_camera_{i}",
                p=[0, 0, self.camera_h],
                q=q,
                width=400,
                height=160,
                near=0.1,
                far=10,
                fov=np.pi / 3,
                actor_uid="mobile_base",
            )
            cameras.append(camera)
        return cameras


class MobilePandaSingleArmDefaultConfig(MobilePandaDualArmDefaultConfig):
    def __init__(self) -> None:
        super().__init__()
        self.urdf_path = "{PACKAGE_ASSET_DIR}/descriptions/mobile_panda_single_arm.urdf"
        self.urdf_config = dict(
            _materials=dict(
                gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
            ),
            link=dict(
                right_panda_leftfinger=dict(
                    material="gripper", patch_radius=0.1, min_patch_radius=0.1
                ),
                right_panda_rightfinger=dict(
                    material="gripper", patch_radius=0.1, min_patch_radius=0.1
                ),
            ),
        )

        self.arm_joint_names = {
            "right": self.arm_joint_names["right"],
        }
        self.gripper_joint_names = {
            "right": self.gripper_joint_names["right"],
        }
        self.ee_link_name = {
            "right": self.ee_link_name["right"],
        }


def test():
    a = MobilePandaDualArmDefaultConfig()
    # print_dict(a.controllers)
    d = a.controllers
    for k, v in d.items():
        print(k, ":")
        print(v)
        print("------------")
