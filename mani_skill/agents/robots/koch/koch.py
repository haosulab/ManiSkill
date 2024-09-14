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
class Koch(BaseAgent):
    scale = 2.0
    uid = "koch-v1.1"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/koch/follower_arm_v1.1_simplified.urdf"
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            link_6=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            gripper=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([0, 0, 0, 0, 0, -1]),
            pose=sapien.Pose(),
        )
    )

    def __init__(self, *args, robot_chassis_colors=[1, 1, 1, 1], **kwargs):
        self.robot_chassis_colors = robot_chassis_colors
        """either a RGBA color or a list of RGBA colors for each robot in each parallel environment to then customize the color of the robot chassis"""
        super().__init__(*args, **kwargs)

    # NOTE (xhinrichsen, stao): Controller is temporary - doesn't resemble real robot
    @property
    def _controller_configs(self):
        pd_joint_pos = PDJointPosControllerConfig(
            [joint.name for joint in self.robot.active_joints],
            lower=None,
            upper=None,
            stiffness=1e3,
            damping=1e2,
            force_limit=100,
            normalize_action=False,
        )
        pd_joint_delta_pos = PDJointPosControllerConfig(
            [joint.name for joint in self.robot.active_joints],
            -0.2,
            0.2,
            stiffness=1e3,
            damping=1e2,
            force_limit=100,
            use_delta=True,
        )

        controller_configs = dict(
            pd_joint_delta_pos=pd_joint_delta_pos,
            pd_joint_pos=pd_joint_pos,
        )
        return deepcopy_dict(controller_configs)

    def _after_loading_articulation(self):
        super()._after_loading_articulation()
        self.finger1_link = self.robot.links_map["gripper"]
        self.finger2_link = self.robot.links_map["link_6"]
        self.tcp = self.robot.links_map["gripper_tcp"]
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

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-1]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold
