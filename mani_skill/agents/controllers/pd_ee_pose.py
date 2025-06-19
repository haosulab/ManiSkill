from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union
from time import time

import numpy as np
import torch
from gymnasium import spaces

from mani_skill.agents.controllers.utils.kinematics import Kinematics
from mani_skill.utils import gym_utils, sapien_utils
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
    quaternion_apply,
    quaternion_multiply,
    matrix_to_euler_angles,
)
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, DriveMode

from .base_controller import ControllerConfig
from .pd_joint_pos import PDJointPosController


class PDEEPosController(PDJointPosController):
    """The PD EE Position controller. NOTE that on the GPU it is assumed the controlled robot is not a merged articulation and is the same across every sub-scene"""

    config: "PDEEPosControllerConfig"
    _target_pose = None

    def _check_gpu_sim_works(self):
        assert (
            self.config.frame == "root_translation"
        ), "currently only translation in the root frame for EE control is supported in GPU sim"

    def _initialize_joints(self):
        self.initial_qpos = None
        super()._initialize_joints()
        if self.device.type == "cuda":
            self._check_gpu_sim_works()
        self.kinematics = Kinematics(
            self.config.urdf_path,
            self.config.ee_link,
            self.articulation,
            self.active_joint_indices,
        )

        self.ee_link = self.kinematics.end_link

        if self.config.root_link_name is not None:
            self.root_link = sapien_utils.get_obj_by_name(
                self.articulation.get_links(), self.config.root_link_name
            )
        else:
            self.root_link = self.articulation.root

    def _initialize_action_space(self):
        low = np.float32(np.broadcast_to(self.config.pos_lower, 3))
        high = np.float32(np.broadcast_to(self.config.pos_upper, 3))
        self.single_action_space = spaces.Box(low, high, dtype=np.float32)

    @property
    def ee_pos(self):
        return self.ee_link.pose.p

    @property
    def ee_pose(self):
        return self.ee_link.pose

    @property
    def ee_pose_at_base(self):
        to_base = self.root_link.pose.inv()
        return to_base * (self.ee_pose)

    def reset(self):
        super().reset()
        if self._target_pose is None:
            self._target_pose = self.ee_pose_at_base
        else:
            # TODO (stao): this is a strange way to mask setting individual batched pose parts
            self._target_pose.raw_pose[self.scene._reset_mask] = (
                self.ee_pose_at_base.raw_pose[self.scene._reset_mask]
            )

    def compute_target_pose(self, prev_ee_pose_at_base, action):
        # Keep the current rotation and change the position
        if self.config.use_delta:
            delta_pose = Pose.create(action)
            if self.config.frame == "root_translation":
                target_pose = delta_pose * prev_ee_pose_at_base
            elif self.config.frame == "body_translation":
                target_pose = prev_ee_pose_at_base * delta_pose
            else:
                raise NotImplementedError(self.config.frame)
        else:
            assert self.config.frame == "root_translation", self.config.frame
            target_pose = Pose.create(action)
        return target_pose

    def set_action(self, action: Array):
        action = self._preprocess_action(action)
        self._step = 0
        self._start_qpos = self.qpos

        if self.config.use_target:
            prev_ee_pose_at_base = self._target_pose
        else:
            prev_ee_pose_at_base = self.ee_pose_at_base

        self._target_pose = self.compute_target_pose(prev_ee_pose_at_base, action)
        pos_only = type(self.config) == PDEEPosControllerConfig
        self._target_qpos = self.kinematics.compute_ik(
            self._target_pose,
            self.articulation.get_qpos(),
            pos_only=pos_only,
            action=action,
            use_delta_ik_solver=self.config.use_delta and not self.config.use_target,
        )
        if self._target_qpos is None:
            self._target_qpos = self._start_qpos
        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)

    def get_state(self) -> dict:
        if self.config.use_target:
            return {"target_pose": self._target_pose.raw_pose}
        return {}

    def set_state(self, state: dict):
        if self.config.use_target:
            target_pose = state["target_pose"]
            self._target_pose = Pose.create_from_pq(
                target_pose[:, :3], target_pose[:, 3:]
            )

    def __repr__(self):
        return f"{self.__class__.__name__}(dof={self.single_action_space.shape[0]}, active_joints={len(self.joints)}, end_link={self.config.ee_link}, joints=({', '.join([x.name for x in self.joints])}))"


# TODO (stao): This config should really inherit the pd joint pos controller config
@dataclass
class PDEEPosControllerConfig(ControllerConfig):
    pos_lower: Union[float, Sequence[float]]
    """Lower bound for position control. If a single float then X, Y, and Z rotations are bounded by this value. Otherwise can be three floats to specify each dimensions bounds"""
    pos_upper: Union[float, Sequence[float]]
    """Upper bound for position control. If a single float then X, Y, and Z rotations are bounded by this value. Otherwise can be three floats to specify each dimensions bounds"""

    # TODO (stao): note stiffness, damping, force limit and friction are properties used by PDJointPos controller, which the PDEEPosController controller inherits from
    # this should be changed as its difficult to figure out how this code is used
    stiffness: Union[float, Sequence[float]]
    damping: Union[float, Sequence[float]]
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0

    ee_link: str = None
    """The name of the end-effector link to control. Note that it does not have to be a end-effector necessarily and could just be any link."""
    urdf_path: str = None
    """Path to the URDF file defining the robot to control."""

    frame: Literal[
        "body_translation",
        "root_translation",
    ] = "root_translation"
    """Choice of frame to use for translational and rotational control of the end-effector. To learn how these work explicitly
    with videos of each one's behavior see https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html#pd-ee-end-effector-pose"""
    root_link_name: Optional[str] = None
    """Optionally set different root link for root translation control (e.g. if root is different than base)"""
    use_delta: bool = True
    """Whether to use delta-action control. If true then actions indicate the delta/change in position via translation and orientation via
    rotation. If false, then actions indicate in the base frame (typically wherever the root link of the robot is) what pose the end effector
    should try and reach via inverse kinematics. """
    use_target: bool = False
    """Whether to use the most recent target end-effector pose for control. If false, actions taken in a chosen frame will be taken
    relative to the instantaneous/current end-effector pose. """
    interpolate: bool = False
    normalize_action: bool = True
    """Whether to normalize each action dimension into a range of [-1, 1]. Normally for most machine learning workflows this is recommended to be kept true."""
    drive_mode: Union[Sequence[DriveMode], DriveMode] = "force"
    controller_cls = PDEEPosController


class PDEEPoseController(PDEEPosController):
    config: "PDEEPoseControllerConfig"

    def _check_gpu_sim_works(self):
        assert (
            self.config.frame == "root_translation:root_aligned_body_rotation"
        ), "currently only translation in the root frame for EE control is supported in GPU sim"

    def _initialize_action_space(self):
        low = np.float32(
            np.hstack(
                [
                    np.broadcast_to(self.config.pos_lower, 3),
                    np.broadcast_to(self.config.rot_lower, 3),
                ]
            )
        )
        high = np.float32(
            np.hstack(
                [
                    np.broadcast_to(self.config.pos_upper, 3),
                    np.broadcast_to(self.config.rot_upper, 3),
                ]
            )
        )
        self.single_action_space = spaces.Box(low, high, dtype=np.float32)

    def _clip_and_scale_action(self, action):
        # NOTE(xiqiang): rotation should be clipped by norm.
        pos_action = gym_utils.clip_and_scale_action(
            action[:, :3], self.action_space_low[:3], self.action_space_high[:3]
        )
        # need to clone here to avoid in place modification of the original action data
        rot_action = action[:, 3:].clone()

        rot_norm = torch.linalg.norm(rot_action, axis=1)
        rot_action[rot_norm > 1] = torch.mul(rot_action, 1 / rot_norm[:, None])[
            rot_norm > 1
        ]
        rot_action = rot_action * self.config.rot_lower
        return torch.hstack([pos_action, rot_action])

    def compute_target_pose(self, prev_ee_pose_at_base: Pose, action):
        if self.config.use_delta:
            delta_pos, delta_rot = action[:, 0:3], action[:, 3:6]
            delta_quat = matrix_to_quaternion(euler_angles_to_matrix(delta_rot, "XYZ"))
            delta_pose = Pose.create_from_pq(delta_pos, delta_quat)
            if "root_aligned_body_rotation" in self.config.frame:
                q = quaternion_multiply(delta_pose.q, prev_ee_pose_at_base.q)
            if "body_aligned_body_rotation" in self.config.frame:
                q = quaternion_multiply(prev_ee_pose_at_base.q, delta_pose.q)
            if "root_translation" in self.config.frame:
                p = prev_ee_pose_at_base.p + delta_pos
            if "body_translation" in self.config.frame:
                p = prev_ee_pose_at_base.p + quaternion_apply(
                    prev_ee_pose_at_base.q, delta_pose.p
                )
            target_pose = Pose.create_from_pq(p, q)
        else:
            assert (
                self.config.frame == "root_translation:root_aligned_body_rotation"
            ), self.config.frame
            target_pos, target_rot = action[:, 0:3], action[:, 3:6]
            target_quat = matrix_to_quaternion(
                euler_angles_to_matrix(target_rot, "XYZ")
            )
            target_pose = Pose.create_from_pq(target_pos, target_quat)

        return target_pose


@dataclass
class PDEEPoseControllerConfig(PDEEPosControllerConfig):

    rot_lower: Union[float, Sequence[float]] = None
    """Lower bound for rotation control. If a single float then X, Y, and Z rotations are bounded by this value. Otherwise can be three floats to specify each dimensions bounds"""
    rot_upper: Union[float, Sequence[float]] = None
    """Upper bound for rotation control. If a single float then X, Y, and Z rotations are bounded by this value. Otherwise can be three floats to specify each dimensions bounds"""
    stiffness: Union[float, Sequence[float]] = None
    damping: Union[float, Sequence[float]] = None
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0

    frame: Literal[
        "body_translation:root_aligned_body_rotation",
        "root_translation:root_aligned_body_rotation",
        "body_translation:body_aligned_body_rotation",
        "root_translation:body_aligned_body_rotation",
    ] = "root_translation:root_aligned_body_rotation"
    """Choice of frame to use for translational and rotational control of the end-effector. To learn how these work explicitly
    with videos of each one's behavior see https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html#pd-ee-end-effector-pose"""

    controller_cls = PDEEPoseController








# ======================================================================================================================
# ======================================================================================================================
# NEW EE_POSE CONTROLLER


class PDEEPoseController_NEW(PDEEPosController):
    config: "PDEEPoseControllerConfig_NEW"

    def _check_gpu_sim_works(self):
        assert (
            self.config.frame == "root_translation:root_aligned_body_rotation"
        ), "currently only translation in the root frame for EE control is supported in GPU sim"

    def _initialize_action_space(self):
        low = np.float32(
            np.hstack(
                [
                    np.broadcast_to(self.config.pos_lower, 3),
                    np.broadcast_to(self.config.rot_lower, 3),
                ]
            )
        )
        high = np.float32(
            np.hstack(
                [
                    np.broadcast_to(self.config.pos_upper, 3),
                    np.broadcast_to(self.config.rot_upper, 3),
                ]
            )
        )
        self.single_action_space = spaces.Box(low, high, dtype=np.float32)

    def _clip_and_scale_action(self, action):
        # NOTE(xiqiang): rotation should be clipped by norm.
        pos_action = gym_utils.clip_and_scale_action(
            action[:, :3], self.action_space_low[:3], self.action_space_high[:3]
        )
        # need to clone here to avoid in place modification of the original action data
        rot_action = action[:, 3:].clone()

        rot_norm = torch.linalg.norm(rot_action, axis=1)
        rot_action[rot_norm > 1] = torch.mul(rot_action, 1 / rot_norm[:, None])[
            rot_norm > 1
        ]
        rot_action = rot_action * self.config.rot_lower
        return torch.hstack([pos_action, rot_action])

    def set_action(self, action: Array):
        """ Action is a 6D pose in the root frame.

        Args:
            action (Array) [B x 6]: 6D pose in the root frame
        """
        t0 = time()
        assert self.config.frame == "root_translation:root_aligned_body_rotation", self.config.frame
        assert action.shape[1] == 6, f"Action must be a 6D pose in the root frame, got {action.shape}"
        assert self.config.use_target, "use_target must be True for the new controller. I don't know what this does and it's not used here."
        action = self._preprocess_action(action)
        B = action.shape[0]

        # Compute the desired change in end effector pose
        target_pos, target_rot = action[:, 0:3], action[:, 3:6]
        target_quat = matrix_to_quaternion(
            euler_angles_to_matrix(target_rot, "XYZ")
        )
        target_pose = Pose.create_from_pq(target_pos, target_quat)
        current_ee_pose = self.ee_pose_at_base
        delta_pose_pose = target_pose * current_ee_pose.inv()
        delta_pose = torch.zeros((B, 6), device=self.device, dtype=torch.float32)
        delta_pose[:, 0:3] = delta_pose_pose.p
        delta_pose[:, 3:6] = matrix_to_euler_angles(delta_pose_pose.to_transformation_matrix()[:, :3, :3], "XYZ")


        # Get the jacobian - [B x 6 x N]. [:, 0:3, :] are the position rows. [:, 3:6, :] are the rotation rows.
        q0 = self.articulation.get_qpos()[:, self.kinematics.active_ancestor_joint_idxs]
        jacobian = self.kinematics.pk_chain.jacobian(q0)
        jacobian = jacobian[:, :, self.kinematics.qmask]
        delta_joint_pos = torch.linalg.pinv(jacobian) @ delta_pose.unsqueeze(-1)
        delta_joint_pos0 = delta_joint_pos.clone().detach()

        # scale down
        norm = torch.linalg.norm(delta_joint_pos.squeeze(-1), dim=1)
        norm_threshold = 0.1
        scale = 0.1
        mask = norm > norm_threshold
        print("mask:", mask, f"\tnorm:", norm)
        delta_joint_pos[mask, :, :] /= norm[mask].view(mask.sum(), 1, 1) * 1/scale


        print()
        print("delta_joint_pos0:", delta_joint_pos0[0, :, 0])
        print("delta_joint_pos: ", delta_joint_pos[0, :, 0])
        q_updated = q0[:, self.kinematics.qmask] + delta_joint_pos.squeeze(-1)


        # self._target_qpos = self.kinematics.compute_ik(
        #     self._target_pose,
        #     self.articulation.get_qpos(),
        #     pos_only=False,
        #     action=action,
        #     use_delta_ik_solver=self.config.use_delta and not self.config.use_target,
        # )
        # t = time()
        assert not self.config.interpolate, "interpolate is not supported in the new controller"
        self.set_drive_targets(q_updated)
        # print(f"Time taken: {time() - t0}")

        # R_difference = torch.tensor(root__T__ee_desired.R * np.linalg.inv(current_ee_pose.R), dtype=torch.float32, device=DEVICE)
        # action_delta[env_idx, 0:3] = torch.tensor(
        #     world__T__ee_desired.t - current_ee_pose.t, dtype=torch.float32, device=DEVICE
        # )
        # action_delta[env_idx, 3:6] = matrix_to_euler_angles(R_difference, "XYZ")
        # action_delta[env_idx, 6] = gripper_angle_desired
        # # Increase the delta to make the tracking more aggressive. 0:6 means leaving the gripper alone
        # action_delta[:, 0:3] *= 7.0  # xyz
        # action_delta[:, 3:6] *= 1.0  # rpy
        # print(f"Time taken: {time() - t0}")





@dataclass
class PDEEPoseControllerConfig_NEW(PDEEPosControllerConfig):

    rot_lower: Union[float, Sequence[float]] = None
    """Lower bound for rotation control. If a single float then X, Y, and Z rotations are bounded by this value. Otherwise can be three floats to specify each dimensions bounds"""
    rot_upper: Union[float, Sequence[float]] = None
    """Upper bound for rotation control. If a single float then X, Y, and Z rotations are bounded by this value. Otherwise can be three floats to specify each dimensions bounds"""
    stiffness: Union[float, Sequence[float]] = None
    damping: Union[float, Sequence[float]] = None
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0

    frame: Literal[
        "body_translation:root_aligned_body_rotation",
        "root_translation:root_aligned_body_rotation",
        "body_translation:body_aligned_body_rotation",
        "root_translation:body_aligned_body_rotation",
    ] = "root_translation:root_aligned_body_rotation"
    """Choice of frame to use for translational and rotational control of the end-effector. To learn how these work explicitly
    with videos of each one's behavior see https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html#pd-ee-end-effector-pose"""

    controller_cls = PDEEPoseController_NEW

    def __post_init__(self):
        assert not self.use_delta, "use_delta is not supported in the new controller"