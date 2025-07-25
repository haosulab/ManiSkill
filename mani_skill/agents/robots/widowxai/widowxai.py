import copy

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
    urdf_path = f"{ASSET_DIR}/robots/widowxai/wxai_base.urdf"
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            gripper_left=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
            gripper_right=dict(
                material="gripper", patch_radius=0.1, min_patch_radius=0.1
            ),
        ),
    )

    keyframes = dict(
        ready_to_grasp=Keyframe(
            qpos=np.array(
                [
                    0.0,
                    1.38,
                    1.04,
                    -1.26,
                    0.0,
                    0.0,
                    0.026,
                    0.026,
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
        "left_carriage_joint",
    ]
    gripper_joint_names = [
        # only control the control joint, not the mimicked one
        # "right_carriage_joint",
        "left_carriage_joint",
    ]
    ee_link_name = "ee_gripper_link"
    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100
    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100

    @property
    def _controller_configs(self):
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
            use_target=False,
        )
        gripper_pd_joint_pos = PDJointPosControllerConfig(
            self.gripper_joint_names,
            lower=0.0,
            upper=0.044,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
        )
        arm_pd_joint_target_delta_pos = copy.deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos,
                gripper=gripper_pd_joint_pos,
            ),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos),
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos,
                gripper=gripper_pd_joint_pos,
            ),
        )
        return deepcopy_dict(controller_configs)

    def _after_loading_articulation(self):
        self.finger1_link = self.robot.links_map["gripper_left"]
        self.finger2_link = self.robot.links_map["gripper_right"]
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose

    def is_grasping(self, object: Actor, min_force=0.2, max_angle=85):
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
        _is_grasped = torch.logical_and(lflag, rflag)
        return _is_grasped

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-2]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold
