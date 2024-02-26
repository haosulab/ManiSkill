import numpy as np
import sapien.physx as physx
from sapien import Pose

from mani_skill2 import PACKAGE_ASSET_DIR
from mani_skill2.utils.sapien_utils import get_obj_by_name, get_objs_by_names

from .mobile_panda_dual_arm import MobilePandaDualArm


class MobilePandaSingleArm(MobilePandaDualArm):
    # single arm robot shares the same controllers and sensor config code
    uid = "mobile_panda_single_arm"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/mobile_panda_single_arm.urdf"
    urdf_config = dict(
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

    def _after_loading_articulation(self):
        self.arm_joint_names = {
            "right": self.arm_joint_names["right"],
        }
        self.gripper_joint_names = {
            "right": self.gripper_joint_names["right"],
        }
        self.ee_link_name = {
            "right": self.ee_link_name["right"],
        }

    def _after_init(self):
        super()._after_init()

        self.finger1_joint, self.finger2_joint = get_objs_by_names(
            self.robot.get_joints(),
            ["right_panda_finger_joint1", "right_panda_finger_joint2"],
        )
        self.finger1_link, self.finger2_link = get_objs_by_names(
            self.robot.get_links(),
            ["right_panda_leftfinger", "right_panda_rightfinger"],
        )
        self.hand: physx.PhysxArticulationLinkComponent = get_obj_by_name(
            self.robot.get_links(), "right_panda_hand"
        )

    def get_fingers_info(self):
        fingers_pos = self.get_ee_coords().flatten()
        fingers_vel = self.get_ee_vels().flatten()
        return {
            "fingers_pos": fingers_pos,
            "fingers_vel": fingers_vel,
        }

    def get_ee_coords(self):
        finger_tips = [
            (self.finger2_joint.get_global_pose() * Pose([0, 0.035, 0])).p,
            (self.finger1_joint.get_global_pose() * Pose([0, -0.035, 0])).p,
        ]
        return np.array(finger_tips)

    def get_ee_vels(self):
        finger_vels = [
            self.finger2_link.get_linear_velocity(),
            self.finger1_link.get_linear_velocity(),
        ]
        return np.array(finger_vels)

    def get_ee_coords_sample(self):
        l = 0.0355
        r = 0.052
        ret = []
        for i in range(10):
            x = (l * i + (4 - i) * r) / 4
            finger_tips = [
                (self.finger2_joint.get_global_pose() * Pose([0, x, 0])).p,
                (self.finger1_joint.get_global_pose() * Pose([0, -x, 0])).p,
            ]
            ret.append(finger_tips)
        return np.array(ret).transpose((1, 0, 2))

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """Build a grasp pose (panda_hand)."""
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return Pose(T)
