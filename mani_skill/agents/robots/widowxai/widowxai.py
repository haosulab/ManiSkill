import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor


@register_agent(asset_download_ids=["widowxai"])
class WidowXAI(BaseAgent):
    uid = "widowxai"
    urdf_path = f"{ASSET_DIR}/robots/widowxai/wxai.urdf"
    urdf_config = dict()

    keyframes = dict(
        ready_to_grasp=Keyframe(
            qpos=np.array(
                [
                    0.0,
                    2.0,
                    1.12,
                    0.7,
                    0.0,
                    0.0,
                    0.04,
                    0.04,
                ]
            ),
            pose=sapien.Pose(),
        )
    )

    arm_joint_names = [
        "joint_0",
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
    ]
    gripper_joint_names = [
        "right_carriage_joint",
        "left_carriage_joint",
    ]
    ee_link_name = "ee_gripper_link"

    def _after_loading_articulation(self):
        self.finger1_link = self.robot.links_map["gripper_left"]
        self.finger2_link = self.robot.links_map["gripper_right"]
        self.tcp = sapien_utils.get_obj_by_name(self.robot.get_links(), self.ee_link_name)

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        """Check if the robot is grasping an object

        Args:
            object (Actor): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        # direction to open the gripper
        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-2]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold