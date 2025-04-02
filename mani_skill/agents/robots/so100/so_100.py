import copy

import numpy as np
import sapien
import sapien.render
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor


@register_agent()
class SO100(BaseAgent):
    uid = "so100"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/so100/SO_5DOF_ARM100_8j/original.urdf"
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2, dynamic_friction=2, restitution=0.0)
        ),
        link=dict(
            Fixed_Jaw=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            Moving_Jaw=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([0, 2.2, 3.017, -0.25, 0, 0.6044]),
            pose=sapien.Pose(),
        ),
        elevated_turn=Keyframe(
            qpos=np.array([0, 2.2, 2.75, -0.25, -np.pi / 2, 1.0]),
            pose=sapien.Pose(),
        ),
        to_push=Keyframe(
            qpos=np.array([0, 2.2, 3.017, -0.25, -np.pi / 2, 0.6044]),
            pose=sapien.Pose(),
        ),
        closed_gripper=Keyframe(
            qpos=np.array([0, 2.2, 3.017, -0.25, -np.pi / 2, 0]),
            pose=sapien.Pose(),
        ),
        zero=Keyframe(
            qpos=np.array([0.0] * 6),
            pose=sapien.Pose(),
        ),
    )

    @property
    def _controller_configs(self):
        pd_joint_pos = PDJointPosControllerConfig(
            [joint.name for joint in self.robot.active_joints],
            lower=None,
            upper=None,
            stiffness=[1e3] * 6,
            damping=[1e2] * 6,
            force_limit=100,
            normalize_action=False,
        )

        pd_joint_delta_pos = PDJointPosControllerConfig(
            [joint.name for joint in self.robot.active_joints],
            -0.1,
            0.1,
            stiffness=[1e3] * 6,
            damping=[1e2] * 6,
            force_limit=100,
            use_delta=True,
            use_target=False,
        )
        pd_joint_target_delta_pos = copy.deepcopy(pd_joint_delta_pos)
        pd_joint_target_delta_pos.use_target = True

        controller_configs = dict(
            pd_joint_delta_pos=pd_joint_delta_pos,
            pd_joint_pos=pd_joint_pos,
            pd_joint_target_delta_pos=pd_joint_target_delta_pos,
        )
        return deepcopy_dict(controller_configs)

    def set_colors(self, base_color=None, motor_color=None):
        """
        basecolor:  RGBA length 4
        motorcolor: RBGA length 4
        """
        if base_color is not None:
            self.robot_chassis_colors = base_color
        if motor_color is not None:
            self.robot_motor_colors = motor_color
        for link in self.robot.links:
            for i, obj in enumerate(link._objs):
                rb_comp = obj.entity.find_component_by_type(
                    sapien.render.RenderBodyComponent
                )
                if rb_comp is not None:
                    rb_comp: sapien.render.RenderBodyComponent
                    meshes_to_modify = [
                        x for x in rb_comp.render_shapes if "chassis" in x.name
                    ]
                    for mesh in meshes_to_modify:
                        if isinstance(self.robot_chassis_colors[0], list):
                            color = self.robot_chassis_colors[i]
                        else:
                            color = self.robot_chassis_colors
                        mesh.material.base_color = color

                    other_meshes_to_modify = [
                        x for x in rb_comp.render_shapes if "motor" in x.name
                    ]
                    for mesh in other_meshes_to_modify:
                        if isinstance(self.robot_motor_colors[0], list):
                            color = self.robot_motor_colors[i]
                        else:
                            color = self.robot_motor_colors
                        for part in mesh.parts:
                            part.material.base_color = color

    def _after_loading_articulation(self):
        super()._after_loading_articulation()
        # self.set_colors()
        self.finger1_link = self.robot.links_map["Fixed_Jaw"]
        self.finger2_link = self.robot.links_map["Moving_Jaw"]
        # self.tcp = self.robot.links_map["gripper_tcp"]
        # self.tcp2 = self.robot.links_map["gripper_tcp2"]
        # self.back_tcp = self.robot.links_map["back_tcp"]

    @property
    def tcp_pos(self):
        # computes the tool center point as the mid point between the the fixed and moving jaw's tips
        fixed_jaw_offset = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        moving_jaw_offset = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        fixed_jaw_pos = self.finger1_link.pose.p + fixed_jaw_offset
        moving_jaw_pos = self.finger2_link.pose.p + moving_jaw_offset
        return (fixed_jaw_pos + moving_jaw_pos) / 2

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=110):
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
