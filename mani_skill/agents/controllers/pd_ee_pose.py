from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence, Union

import numpy as np
import torch
from gymnasium import spaces

from mani_skill.agents.controllers.utils.kinematics import Kinematics
from mani_skill.utils import gym_utils, sapien_utils
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    quaternion_apply,
    quaternion_multiply,
    quaternion_to_matrix,
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
        if self.config.use_target:
            if self._target_pose is None:
                self._target_pose = self.ee_pose_at_base
            else:
                # TODO (stao): this is a strange way to mask setting individual batched pose parts
                self._target_pose.raw_pose[
                    self.scene._reset_mask
                ] = self.ee_pose_at_base.raw_pose[self.scene._reset_mask]

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

        # we only need to use the target pose for CPU sim or if a virtual target is enabled
        # if we have no virtual target and using the gpu sim we can directly use the given action without
        # having to recompute the new target pose based on the action delta.
        ik_via_target_pose = self.config.use_target or not self.scene.gpu_sim_enabled
        if ik_via_target_pose:
            self._target_pose = self.compute_target_pose(prev_ee_pose_at_base, action)
        pos_only = type(self.config) == PDEEPosControllerConfig
        if pos_only:
            action = torch.hstack([action, torch.zeros(action.shape[0], 3, device=self.device)])

        self._target_qpos = self.kinematics.compute_ik(
            pose=self._target_pose if ik_via_target_pose else action,
            q0=self.articulation.get_qpos(),
            is_delta_pose=not ik_via_target_pose and self.config.use_delta,
            current_pose=self.ee_pose_at_base,
            solver_config=self.config.delta_solver_config,
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
    delta_solver_config: dict = field(
        default_factory=lambda: dict(type="levenberg_marquardt", alpha=1.0)
    )
    """Configuration for the delta IK solver. Default is `dict(type="levenberg_marquardt", alpha=1.0)`. type can be one of "levenberg_marquardt" or "pseudo_inverse". alpha is a scaling term applied to the delta joint positions generated by the solver. Generally levenberg_marquardt is faster and more accurate than pseudo_inverse and is the recommended option, see https://github.com/haosulab/ManiSkill/issues/955#issuecomment-2742253342 for some analysis on performance."""
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
