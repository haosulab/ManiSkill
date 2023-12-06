import numpy as np
import sapien
import sapien.physx as physx

from mani_skill2.agents.base_agent import BaseAgent
from mani_skill2.agents.robots.xmate3.configs import Xmate3RobotiqDefaultConfig
from mani_skill2.utils.common import compute_angle_between
from mani_skill2.utils.sapien_utils import (
    compute_total_impulse,
    get_actor_contacts,
    get_obj_by_name,
    get_pairwise_contact_impulse,
)


class Xmate3Robotiq(BaseAgent):
    config: Xmate3RobotiqDefaultConfig

    @classmethod
    def get_default_config(cls):
        return Xmate3RobotiqDefaultConfig()

    def _after_init(self):
        self.finger1_link: sapien.Entity = get_obj_by_name(
            self.robot.get_links(), "left_inner_finger_pad"
        ).entity
        self.finger2_link: sapien.Entity = get_obj_by_name(
            self.robot.get_links(), "right_inner_finger_pad"
        ).entity
        self.tcp: physx.PhysxArticulationLinkComponent = get_obj_by_name(
            self.robot.get_links(), self.config.ee_link_name
        )

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
