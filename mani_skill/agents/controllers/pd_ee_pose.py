from dataclasses import dataclass
from typing import List, Sequence, Union

import fast_kinematics
import numpy as np
import sapien.physx as physx
import torch
from gymnasium import spaces

from mani_skill import logger
from mani_skill.utils import common, gym_utils, sapien_utils
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
from mani_skill.utils.structs import ArticulationJoint, Pose
from mani_skill.utils.structs.types import Array, DriveMode

from .base_controller import ControllerConfig
from .pd_joint_pos import PDJointPosController


# NOTE(jigu): not necessary to inherit, just for convenience
class PDEEPosController(PDJointPosController):
    config: "PDEEPosControllerConfig"
    _target_pose = None

    def _initialize_joints(self):
        self.initial_qpos = None
        super()._initialize_joints()
        if self.config.ee_link:
            self.ee_link = sapien_utils.get_obj_by_name(
                self.articulation.get_links(), self.config.ee_link
            )
        else:
            # The child link of last joint is assumed to be the end-effector.
            self.ee_link = self.joints[-1].get_child_link()
            logger.warn(
                "Configuration did not define a ee_link name, using the child link of the last joint"
            )
        self.ee_link_idx = self.articulation.get_links().index(self.ee_link)

        if physx.is_gpu_enabled():
            self.fast_kinematics_model = fast_kinematics.FastKinematics(
                self.config.urdf_path, self.scene.num_envs, self.config.ee_link
            )
            # note that everything past the end-effector is ignored. Any joint whose ancestor is self.ee_link is ignored
            # get_joints returns the joints in level order
            # for joint in joints
            cur_link = self.ee_link.joint.parent_link
            active_ancestor_joints: List[ArticulationJoint] = []
            while cur_link is not None:
                if cur_link.joint.active_index is not None:
                    active_ancestor_joints.append(cur_link.joint)
                cur_link = cur_link.joint.parent_link
            active_ancestor_joints = active_ancestor_joints[::-1]
            self.active_ancestor_joints = active_ancestor_joints
            # initially self.active_joint_indices references active joints that are controlled.
            self.active_ancestor_joint_idxs = [
                x.active_index for x in self.active_ancestor_joints
            ]
            controlled_joints_idx_in_qmask = [
                self.active_ancestor_joint_idxs.index(idx)
                for idx in self.active_joint_indices
            ]
            self.qmask = torch.zeros(
                len(self.active_ancestor_joints), dtype=bool, device=self.device
            )
            self.qmask[controlled_joints_idx_in_qmask] = 1
        else:
            self.qmask = torch.zeros(
                self.articulation.max_dof, dtype=bool, device=self.device
            )
            self.pmodel = self.articulation._objs[0].create_pinocchio_model()
            self.qmask[self.active_joint_indices] = 1

    def _initialize_action_space(self):
        low = np.float32(np.broadcast_to(self.config.lower, 3))
        high = np.float32(np.broadcast_to(self.config.upper, 3))
        self.single_action_space = spaces.Box(low, high, dtype=np.float32)

    @property
    def ee_pos(self):
        return self.ee_link.pose.p

    @property
    def ee_pose(self):
        return self.ee_link.pose

    @property
    def ee_pose_at_base(self):
        to_base = self.articulation.pose.inv()
        return to_base * (self.ee_pose)

    def reset(self):
        super().reset()
        if self._target_pose is None:
            self._target_pose = self.ee_pose_at_base
        else:
            # TODO (stao): this is a strange way to mask setting individual batched pose parts
            self._target_pose.raw_pose[
                self.scene._reset_mask
            ] = self.ee_pose_at_base.raw_pose[self.scene._reset_mask]

    def compute_ik(
        self, target_pose: Pose, action: Array, pos_only=True, max_iterations=100
    ):
        # Assume the target pose is defined in the base frame
        if physx.is_gpu_enabled():
            jacobian = (
                self.fast_kinematics_model.jacobian_mixed_frame_pytorch(
                    self.articulation.get_qpos()[:, self.active_ancestor_joint_idxs]
                )
                .view(-1, len(self.active_ancestor_joints), 6)
                .permute(0, 2, 1)
            )
            jacobian = jacobian[:, :, self.qmask]
            # NOTE (stao): a bit of a hacky way to check if we want to do IK on position or pose here
            if pos_only:
                jacobian = jacobian[:, 0:3]

            # NOTE (stao): this method of IK is from https://mathweb.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf by Samuel R. Buss
            delta_joint_pos = torch.linalg.pinv(jacobian) @ action.unsqueeze(-1)
            return self.qpos + delta_joint_pos.squeeze(-1)

        else:
            result, success, error = self.pmodel.compute_inverse_kinematics(
                self.ee_link_idx,
                target_pose.sp,
                initial_qpos=common.to_numpy(self.articulation.get_qpos()).squeeze(0),
                active_qmask=self.qmask,
                max_iterations=max_iterations,
            )
        if success:
            return common.to_tensor([result[self.active_joint_indices]])
        else:
            return None

    def compute_target_pose(self, prev_ee_pose_at_base, action):
        # Keep the current rotation and change the position
        if self.config.use_delta:
            delta_pose = Pose.create(action)

            if self.config.frame == "base":
                target_pose = delta_pose * prev_ee_pose_at_base
            elif self.config.frame == "ee":
                target_pose = prev_ee_pose_at_base * delta_pose
            else:
                raise NotImplementedError(self.config.frame)
        else:
            assert self.config.frame == "base", self.config.frame
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
        self._target_qpos = self.compute_ik(self._target_pose, action)
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
        # if self.config.use_target:
        #     target_pose = state["target_pose"]
        #     self._target_pose = sapien.Pose(target_pose[:3], target_pose[3:])
        raise NotImplementedError()


@dataclass
class PDEEPosControllerConfig(ControllerConfig):
    lower: Union[float, Sequence[float]]
    upper: Union[float, Sequence[float]]
    stiffness: Union[float, Sequence[float]]
    damping: Union[float, Sequence[float]]
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0
    ee_link: str = None
    urdf_path: str = None
    frame: str = "ee"  # [base, ee]
    use_delta: bool = True
    use_target: bool = False
    interpolate: bool = False
    normalize_action: bool = True
    drive_mode: Union[Sequence[DriveMode], DriveMode] = "force"
    controller_cls = PDEEPosController


class PDEEPoseController(PDEEPosController):
    config: "PDEEPoseControllerConfig"

    def _initialize_action_space(self):
        low = np.float32(
            np.hstack(
                [
                    np.broadcast_to(self.config.pos_lower, 3),
                    np.broadcast_to(-self.config.rot_bound, 3),
                ]
            )
        )
        high = np.float32(
            np.hstack(
                [
                    np.broadcast_to(self.config.pos_upper, 3),
                    np.broadcast_to(self.config.rot_bound, 3),
                ]
            )
        )
        self.single_action_space = spaces.Box(low, high, dtype=np.float32)

    def _clip_and_scale_action(self, action):
        # NOTE(xiqiang): rotation should be clipped by norm.
        pos_action = gym_utils.clip_and_scale_action(
            action[:, :3], self.action_space_low[:3], self.action_space_high[:3]
        )
        rot_action = action[:, 3:]

        rot_norm = torch.linalg.norm(rot_action, axis=1)
        rot_action[rot_norm > 1] = torch.mul(rot_action, 1 / rot_norm[:, None])[
            rot_norm > 1
        ]
        rot_action = rot_action * self.config.rot_bound
        return torch.hstack([pos_action, rot_action])

    def compute_ik(self, target_pose: Pose, action: Array, max_iterations=100):
        return super().compute_ik(
            target_pose, action, pos_only=False, max_iterations=max_iterations
        )

    def compute_target_pose(self, prev_ee_pose_at_base: Pose, action):
        if self.config.use_delta:
            delta_pos, delta_rot = action[:, 0:3], action[:, 3:6]
            delta_quat = matrix_to_quaternion(euler_angles_to_matrix(delta_rot, "XYZ"))
            delta_pose = Pose.create_from_pq(delta_pos, delta_quat)

            if self.config.frame == "base":
                target_pose = delta_pose * prev_ee_pose_at_base
            elif self.config.frame == "ee":
                target_pose = prev_ee_pose_at_base * delta_pose
            elif self.config.frame == "ee_align":
                # origin at ee but base rotation
                target_pose = delta_pose * prev_ee_pose_at_base
                target_pose.set_p(prev_ee_pose_at_base.p + delta_pos)
            else:
                raise NotImplementedError(self.config.frame)
        else:
            assert self.config.frame == "base", self.config.frame
            target_pos, target_rot = action[:, 0:3], action[:, 3:6]
            target_quat = matrix_to_quaternion(
                euler_angles_to_matrix(target_rot, "XYZ")
            )
            # target_quat = Rotation.from_rotvec(target_rot).as_quat()[[3, 0, 1, 2]]
            target_pose = Pose.create_from_pq(target_pos, target_quat)

        return target_pose


@dataclass
class PDEEPoseControllerConfig(ControllerConfig):
    pos_lower: Union[float, Sequence[float]]
    pos_upper: Union[float, Sequence[float]]
    rot_bound: float
    stiffness: Union[float, Sequence[float]]
    damping: Union[float, Sequence[float]]
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0
    ee_link: str = None
    urdf_path: str = None
    frame: str = "ee"  # [base, ee, ee_align]
    use_delta: bool = True
    use_target: bool = False
    interpolate: bool = False
    normalize_action: bool = True
    drive_mode: Union[Sequence[DriveMode], DriveMode] = "force"
    controller_cls = PDEEPoseController
