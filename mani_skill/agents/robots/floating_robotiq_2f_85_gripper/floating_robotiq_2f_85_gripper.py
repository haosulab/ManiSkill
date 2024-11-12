from typing import Dict, List, Union

import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR, PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, DictControllerConfig, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.controllers.base_controller import ControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor


@register_agent(asset_download_ids=["robotiq_2f"])
class FloatingRobotiq2F85Gripper(BaseAgent):
    uid = "floating_robotiq_2f_85_gripper"
    urdf_path = f"{ASSET_DIR}/robots/robotiq_2f/floating_robotiq_2f_85.urdf"
    disable_self_collisions = True
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
    keyframes = dict(
        open_facing_up=Keyframe(
            qpos=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            pose=sapien.Pose(p=np.array([0.0, 0.0, 0.5])),
        ),
        open_facing_side=Keyframe(
            qpos=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            pose=sapien.Pose(
                p=np.array([0.0, 0.0, 0.5]), q=np.array([0.7071, 0, 0.7071, 0])
            ),
        ),
    )
    root_joint_names = [
        "root_x_axis_joint",
        "root_y_axis_joint",
        "root_z_axis_joint",
        "root_x_rot_joint",
        "root_y_rot_joint",
        "root_z_rot_joint",
    ]
    ee_link_name = "left_inner_finger_pad"

    @property
    def _controller_configs(
        self,
    ) -> Dict[str, Union[ControllerConfig, DictControllerConfig]]:

        # define a simple controller to control the floating base with XYZ/RPY control.
        base_pd_joint_pos = PDJointPosControllerConfig(
            joint_names=self.root_joint_names,
            lower=None,
            upper=None,
            stiffness=1e3,
            damping=1e2,
            force_limit=100,
            normalize_action=False,
        )
        base_pd_joint_delta_pos = PDJointPosControllerConfig(
            joint_names=self.root_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=1e3,
            damping=1e2,
            force_limit=100,
            use_delta=True,
        )

        # define a passive controller config to simply "turn off" other joints from being controlled and set their properties (damping/friction) to 0.
        # these joints are controlled passively by the mimic controller later on.
        passive_finger_joint_names = [
            "left_inner_knuckle_joint",
            "right_inner_knuckle_joint",
            "left_inner_finger_joint",
            "right_inner_finger_joint",
        ]
        passive_finger_joints = PassiveControllerConfig(
            joint_names=passive_finger_joint_names,
            damping=0,
            friction=0,
        )

        finger_joint_names = ["left_outer_knuckle_joint", "right_outer_knuckle_joint"]
        # use a mimic controller config to define one action to control both fingers
        finger_mimic_pd_joint_pos = PDJointPosMimicControllerConfig(
            joint_names=finger_joint_names,
            lower=None,
            upper=None,
            stiffness=1e5,
            damping=1e3,
            force_limit=0.1,
            friction=0.05,
            normalize_action=False,
        )
        finger_mimic_pd_joint_delta_pos = PDJointPosMimicControllerConfig(
            joint_names=finger_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=1e5,
            damping=1e3,
            force_limit=0.1,
            normalize_action=True,
            friction=0.05,
            use_delta=True,
        )
        return dict(
            pd_joint_pos=dict(
                base=base_pd_joint_pos,
                finger=finger_mimic_pd_joint_pos,
                passive_finger_joints=passive_finger_joints,
            ),
            pd_joint_delta_pos=dict(
                base=base_pd_joint_delta_pos,
                finger=finger_mimic_pd_joint_delta_pos,
                passive_finger_joints=passive_finger_joints,
            ),
        )

    def _after_loading_articulation(self):
        outer_finger = self.robot.active_joints_map["right_inner_finger_joint"]
        inner_knuckle = self.robot.active_joints_map["right_inner_knuckle_joint"]
        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()

        # the next 4 magic arrays come from https://github.com/haosulab/cvpr-tutorial-2022/blob/master/debug/robotiq.py which was
        # used to precompute these poses for drive creation
        p_f_right = [-1.6048949e-08, 3.7600022e-02, 4.3000020e-02]
        p_p_right = [1.3578170e-09, -1.7901104e-02, 6.5159947e-03]
        p_f_left = [-1.8080145e-08, 3.7600014e-02, 4.2999994e-02]
        p_p_left = [-1.4041154e-08, -1.7901093e-02, 6.5159872e-03]

        right_drive = self.scene.create_drive(
            lif, sapien.Pose(p_f_right), pad, sapien.Pose(p_p_right)
        )
        right_drive.set_limit_x(0, 0)
        right_drive.set_limit_y(0, 0)
        right_drive.set_limit_z(0, 0)

        outer_finger = self.robot.active_joints_map["left_inner_finger_joint"]
        inner_knuckle = self.robot.active_joints_map["left_inner_knuckle_joint"]
        pad = outer_finger.get_child_link()
        lif = inner_knuckle.get_child_link()

        left_drive = self.scene.create_drive(
            lif, sapien.Pose(p_f_left), pad, sapien.Pose(p_p_left)
        )
        left_drive.set_limit_x(0, 0)
        left_drive.set_limit_y(0, 0)
        left_drive.set_limit_z(0, 0)


    def _after_init(self):
        self.finger1_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_inner_finger"
        )
        self.finger2_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "right_inner_finger"
        )
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )


    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
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
        return torch.logical_and(lflag, rflag)


    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-1]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold
