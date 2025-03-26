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
    # urdf_config = dict(
    #     _materials=dict(
    #         gripper=dict(static_friction=0.3, dynamic_friction=0.3, restitution=0.0)
    #     ),
    #     link=dict(
    #         link_6=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
    #         gripper=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
    #     ),
    # )

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
            stiffness=[1e3] * 5 + [1e2],
            damping=[1e2] * 5 + [1e0],
            force_limit=100,
            normalize_action=False,
        )

        pd_joint_delta_pos = PDJointPosControllerConfig(
            [joint.name for joint in self.robot.active_joints],
            [-0.05, -0.05, -0.05, -0.05, -0.1, -0.05],
            [0.05, 0.05, 0.05, 0.05, 0.1, 0.05],
            stiffness=[123, 50, 102.68, 145, 108.37, 93.3],
            damping=[15.85, 6, 15.34, 16, 16.31, 16.3],
            force_limit=100,
            use_delta=True,
            use_target=True,
        )

        controller_configs = dict(
            pd_joint_delta_pos=pd_joint_delta_pos,
            pd_joint_pos=pd_joint_pos,
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

    # def _after_loading_articulation(self):
    #     super()._after_loading_articulation()
    #     self.set_colors()
    #     self.finger1_link = self.robot.links_map["gripper"]
    #     self.finger2_link = self.robot.links_map["link_6"]
    #     self.tcp = self.robot.links_map["gripper_tcp"]
    #     self.tcp2 = self.robot.links_map["gripper_tcp2"]
    #     self.back_tcp = self.robot.links_map["back_tcp"]
