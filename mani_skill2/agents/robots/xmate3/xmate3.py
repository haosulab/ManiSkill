import numpy as np
import sapien
import sapien.physx as physx

from mani_skill2 import ASSET_DIR
from mani_skill2.agents.base_agent import BaseAgent
from mani_skill2.agents.controllers import *
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.common import compute_angle_between
from mani_skill2.utils.sapien_utils import (
    compute_total_impulse,
    get_actor_contacts,
    get_obj_by_name,
    get_pairwise_contact_impulse,
)


class Xmate3Robotiq(BaseAgent):
    uid = "xmate3_robotiq"
    urdf_path = f"{ASSET_DIR}/robots/xmate3_robotiq/xmate3_robotiq.urdf"
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            left_inner_finger_pad=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            right_inner_finger_pad=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    def __init__(
        self,
        scene: sapien.Scene,
        control_freq: int,
        control_mode: str = None,
        fix_root_link=True,
    ):
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
        super().__init__(scene, control_freq, control_mode, fix_root_link)

    def _after_init(self):
        self.finger1_link: sapien.Entity = get_obj_by_name(
            self.robot.get_links(), "left_inner_finger_pad"
        ).entity
        self.finger2_link: sapien.Entity = get_obj_by_name(
            self.robot.get_links(), "right_inner_finger_pad"
        ).entity
        self.tcp: physx.PhysxArticulationLinkComponent = get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

    @property
    def controller_configs(self):
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

        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            ee_link=self.ee_link_name,
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

    sensor_configs = [
        CameraConfig(
            uid="base_camera",
            p=[0.0, 0.0, 0.0],
            q=[1, 0, 0, 0],
            width=128,
            height=128,
            fov=1.5707,
            near=0.01,
            far=10,
            entity_uid="camera_base_link",
            hide_link=False,
        ),
        CameraConfig(
            uid="hand_camera",
            p=[0.0, 0.0, 0.0],
            q=[1, 0, 0, 0],
            width=128,
            height=128,
            fov=1.5707,
            near=0.01,
            far=10,
            entity_uid="camera_hand_link",
            hide_link=False,
        ),
    ]

    def is_grasping(self, object: sapien.Entity = None, min_impulse=1e-6, max_angle=85):
        contacts = self.scene.get_contacts()
        if object is None:
            finger1_contacts = get_actor_contacts(contacts, self.finger1_link)
            finger2_contacts = get_actor_contacts(contacts, self.finger2_link)
            return (
                np.linalg.norm(compute_total_impulse(finger1_contacts)) >= min_impulse
                and np.linalg.norm(compute_total_impulse(finger2_contacts))
                >= min_impulse
            )
        else:
            limpulse = get_pairwise_contact_impulse(contacts, self.finger1_link, object)
            rimpulse = get_pairwise_contact_impulse(contacts, self.finger2_link, object)

            # direction to open the gripper
            ldirection = self.finger1_link.pose.to_transformation_matrix()[:3, 1]
            rdirection = -self.finger2_link.pose.to_transformation_matrix()[:3, 1]

            # angle between impulse and open direction
            langle = compute_angle_between(ldirection, limpulse)
            rangle = compute_angle_between(rdirection, rimpulse)

            lflag = (
                np.linalg.norm(limpulse) >= min_impulse
                and np.rad2deg(langle) <= max_angle
            )
            rflag = (
                np.linalg.norm(rimpulse) >= min_impulse
                and np.rad2deg(rangle) <= max_angle
            )

            return all([lflag, rflag])

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(approaching, closing)
        T = np.eye(4)
        T[:3, :3] = np.stack([approaching, closing, ortho], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)
