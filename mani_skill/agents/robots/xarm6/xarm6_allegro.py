from copy import deepcopy
from typing import List

import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
import numpy as np
import sapien.core as sapien
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import vectorize_pose

@register_agent()
class XArm6AllegroRight(BaseAgent):
    uid = "xarm6_allegro_right"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/xarm6/xarm6_allegro_right.urdf"
    
    disable_self_collisions = False

    urdf_config = dict(
        _materials=dict(
            tip=dict(static_friction=2.0, dynamic_friction=1.0, restitution=0.0)
        ),
        link={
            "link_3.0_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "link_7.0_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "link_11.0_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
            "link_15.0_tip": dict(
                material="tip", patch_radius=0.1, min_patch_radius=0.1
            ),
        },
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [1.56280772e-03, -1.10912404e+00, -9.71343926e-02, 1.52969832e-04, 1.20606723e+00, 1.66234924e-03,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        zeros=Keyframe(
            qpos=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        stretch_j1=Keyframe(
            qpos=np.array([np.pi/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        stretch_j2=Keyframe(
            qpos=np.array([0, np.pi/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        stretch_j3=Keyframe(
            qpos=np.array([0, 0, np.pi/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        stretch_j4=Keyframe(
            qpos=np.array([0, 0, 0, np.pi/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        stretch_j5=Keyframe(
            qpos=np.array([0, 0, 0, 0, np.pi/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        stretch_j6=Keyframe(
            qpos=np.array([0, 0, 0, 0, 0, np.pi/2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            pose=sapien.Pose([0, 0, 0]),
        ),
        palm_up=Keyframe(
            qpos=np.array(
                [0, 0, 0, 0, 0, 0,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                ]
            ),
            pose=sapien.Pose([0, 0, 0.5], q=[-0.707, 0, 0.707, 0]),
        )
    )

    arm_joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ]
    gripper_joint_names = [
        "joint_0.0",
        "joint_1.0",
        "joint_2.0",
        "joint_3.0",
        "joint_4.0",
        "joint_5.0",
        "joint_6.0",
        "joint_7.0",
        "joint_8.0",
        "joint_9.0",
        "joint_10.0",
        "joint_11.0",
        "joint_12.0",
        "joint_13.0",
        "joint_14.0",
        "joint_15.0",
    ]

    arm_stiffness = 1000
    arm_damping = 50 # [50, 50, 50, 50, 50, 50]
    arm_friction = 0.1 # [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    arm_force_limit = 100

    gripper_stiffness = 4e2
    gripper_damping = 1e1
    gripper_force_limit = 5e1

    # Order for left hand: thumb finger, index finger, middle finger, ring finger
    # Order for right hand: thumb finger, ring finger, middle finger, index finger 
    tip_link_names = [
        "link_15.0_tip",
        "link_11.0_tip",
        "link_7.0_tip",
        "link_3.0_tip",
    ]

    palm_link_name = "palm"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            friction=self.arm_friction,
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
            friction=self.arm_friction,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # PD ee position
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.palm_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.palm_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=None,
            pos_upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link=self.palm_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,
            normalize_action=False,
        )

        arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
        arm_pd_ee_target_delta_pos.use_target = True
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # PD joint velocity
        arm_pd_joint_vel = PDJointVelControllerConfig(
            self.arm_joint_names,
            -1.0,
            1.0,
            self.arm_damping,  # this might need to be tuned separately
            self.arm_force_limit,
            self.arm_friction,
        )

        # PD joint position and velocity
        arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            self.arm_friction,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            friction=self.arm_friction,
            use_delta=True,
        )

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        gripper_pd_joint_pos = PDJointPosControllerConfig(
            self.gripper_joint_names,
            None,
            None,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            normalize_action=False,
        )
        gripper_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.gripper_joint_names,
            -0.1,
            0.1,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            use_delta=True,
        )
        gripper_pd_joint_target_delta_pos = deepcopy(gripper_pd_joint_delta_pos)
        gripper_pd_joint_target_delta_pos.use_target = True
        
        # PD joint velocity
        gripper_pd_joint_vel = PDJointVelControllerConfig(
            self.gripper_joint_names,
            -1.0,
            1.0,
            self.gripper_damping,
            self.gripper_force_limit,
        )

        # PD joint position and velocity
        gripper_pd_joint_pos_vel = PDJointPosVelControllerConfig(
            self.gripper_joint_names,
            None,
            None,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            normalize_action=False,
        )
        gripper_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
            self.gripper_joint_names,
            -0.1,
            0.1,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            use_delta=True,
        )

        # -------------------------------------------------------------------------- #
        # Arm + Gripper Controller Configs
        # -------------------------------------------------------------------------- #
        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_delta_pos
            ),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose, gripper=gripper_pd_joint_pos
            ),
            pd_ee_pose=dict(arm=arm_pd_ee_pose, gripper=gripper_pd_joint_pos),
            # TODO(jigu): how to add boundaries for the following controllers
            pd_joint_target_delta_pos=dict(
                arm=arm_pd_joint_target_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_ee_target_delta_pos=dict(
                arm=arm_pd_ee_target_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose, gripper=gripper_pd_joint_pos
            ),
            pd_joint_vel=dict(arm=arm_pd_joint_vel, gripper=gripper_pd_joint_vel),
            pd_joint_pos_vel=dict(
                arm=arm_pd_joint_pos_vel, gripper=gripper_pd_joint_pos_vel
            ),
            pd_joint_delta_pos_vel=dict(
                arm=arm_pd_joint_delta_pos_vel, gripper=gripper_pd_joint_delta_pos_vel
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)


    def get_proprioception(self):
        """
        Get the proprioceptive state of the agent.
        """
        obs = super().get_proprioception()
        obs.update(
            {
                "palm_pose": self.palm_pose,
                "tip_poses": self.tip_poses.reshape(-1, len(self.tip_links) * 7),
            }
        )

        return obs

    @property
    def tip_poses(self):
        """
        Get the tip pose for each of the finger, four fingers in total
        """
        tip_poses = [
            vectorize_pose(link.pose, device=self.device) for link in self.tip_links
        ]
        return torch.stack(tip_poses, dim=-2)

    @property
    def palm_pose(self):
        """
        Get the palm pose for allegro hand
        """
        return vectorize_pose(self.palm_link.pose, device=self.device)


    def _after_init(self):
        self.tip_links: List[sapien.Entity] = sapien_utils.get_objs_by_names(
            self.robot.get_links(), self.tip_link_names
        )
        self.palm_link: sapien.Entity = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.palm_link_name
        )
        self.tcp = self.palm_link


    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        thumb_contact_forces = self.scene.get_pairwise_contact_forces(
            self.tip_links[0], object
        )
        finger1_contact_forces = self.scene.get_pairwise_contact_forces(
            self.tip_links[1], object
        )
        finger2_contact_forces = self.scene.get_pairwise_contact_forces(
            self.tip_links[2], object
        )
        finger3_contact_forces = self.scene.get_pairwise_contact_forces(
            self.tip_links[3], object
        )

        thumb_force = torch.linalg.norm(thumb_contact_forces, axis=1)
        finger1_force = torch.linalg.norm(finger1_contact_forces, axis=1)
        finger2_force = torch.linalg.norm(finger2_contact_forces, axis=1)
        finger3_force = torch.linalg.norm(finger3_contact_forces, axis=1)

        thumb_direction = self.tip_links[0].pose.to_transformation_matrix()[..., :3, 1]
        finger1_direction = self.tip_links[1].pose.to_transformation_matrix()[..., :3, 1]
        finger2_direction = self.tip_links[2].pose.to_transformation_matrix()[..., :3, 1]
        finger3_direction = self.tip_links[3].pose.to_transformation_matrix()[..., :3, 1]
        
        thumb_angle = common.compute_angle_between(thumb_direction, thumb_contact_forces)
        finger1_angle = common.compute_angle_between(finger1_direction, finger1_contact_forces)
        finger2_angle = common.compute_angle_between(finger2_direction, finger2_contact_forces)
        finger3_angle = common.compute_angle_between(finger3_direction, finger3_contact_forces)

        thumb_flag = torch.logical_and(
            thumb_force >= min_force, torch.rad2deg(thumb_angle) <= max_angle
        )
        finger1_flag = torch.logical_and(
            finger1_force >= min_force, torch.rad2deg(finger1_angle) <= max_angle
        )
        finger2_flag = torch.logical_and(
            finger2_force >= min_force, torch.rad2deg(finger2_angle) <= max_angle
        )
        finger3_flag = torch.logical_and(
            finger3_force >= min_force, torch.rad2deg(finger3_angle) <= max_angle
        )
        return torch.logical_and(thumb_flag, torch.logical_and(finger1_flag, torch.logical_and(finger2_flag, finger3_flag)))


    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :-1]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold


@register_agent()
class XArm6AllegroLeft(XArm6AllegroRight):
    uid = "xarm6_allegro_left"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/xarm6/xarm6_allegro_left.urdf"


@register_agent()
class XArm6AllegroRightWristCamera(XArm6AllegroRight):
    uid = "xarm6_allegro_right_wristcam"

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="hand_camera",
                pose=sapien.Pose(p=[-0.025, 0, 0.025], q=[0.70710678, 0, -0.70710678, 0]),
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["base_link"],
            )
        ]
    
@register_agent()
class XArm6AllegroLeftWristCamera(XArm6AllegroLeft):
    uid = "xarm6_allegro_left_wristcam"

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="hand_camera",
                pose=sapien.Pose(p=[-0.025, 0, 0.025], q=[0.70710678, 0, -0.70710678, 0]),
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["base_link"],
            )
        ]