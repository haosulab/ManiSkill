from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import numpy as np
import sapien.physx as physx
import torch

from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.base import BaseStruct
from mani_skill.utils.structs.decorators import before_gpu_init
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.types import Array

if TYPE_CHECKING:
    from mani_skill.utils.structs.articulation import Articulation


@dataclass
class Joint(BaseStruct[physx.PhysxArticulationJoint]):
    """
    Wrapper around physx.PhysxArticulationJoint objects

    At the moment, all of the same joints across all sub scenes are restricted to having the same properties as each other
    """

    articulation: Articulation
    index: int
    """index of this joint among all joints"""
    active_index: int
    """index of this joint amongst the active joints"""

    child_link: Link = None
    parent_link: Link = None
    name: str = None

    def __hash__(self):
        return self._objs[0].__hash__()

    @classmethod
    def create(
        cls,
        physx_joints: List[physx.PhysxArticulationJoint],
        articulation: Articulation,
        joint_index: int,
        active_joint_index: int,
    ):
        # naming convention of the original physx joints is "scene-<id>-<articulation_name>_<original_joint_name>"
        shared_name = "_".join(
            physx_joints[0].name.replace(articulation.name, "", 1).split("_")[1:]
        )
        child_link = None
        parent_link = None
        if not articulation._merged:
            if physx_joints[0].child_link is not None:
                child_link = articulation.link_map[
                    "_".join(
                        physx_joints[0]
                        .child_link.name.replace(articulation.name, "", 1)
                        .split("_")[1:]
                    )
                ]
            if physx_joints[0].parent_link is not None:
                parent_link = articulation.link_map[
                    "_".join(
                        physx_joints[0]
                        .parent_link.name.replace(articulation.name, "", 1)
                        .split("_")[1:]
                    )
                ]
        return cls(
            articulation=articulation,
            index=joint_index,
            active_index=active_joint_index,
            _objs=physx_joints,
            _scene=articulation._scene,
            _scene_idxs=articulation._scene_idxs,
            child_link=child_link,
            parent_link=parent_link,
            name=shared_name,
        )

    # -------------------------------------------------------------------------- #
    # Functions from physx.PhysxArticulationJoint
    # -------------------------------------------------------------------------- #
    # def get_armature(self) -> numpy.ndarray[numpy.float32, _Shape[m, 1]]: ...
    def get_child_link(self):
        return self.child_link

    def get_damping(self) -> float:
        return self.damping

    def get_dof(self) -> int:
        return self.dof

    def get_drive_mode(self):
        return self.drive_mode

    def get_drive_target(self):
        return self.drive_target

    # def get_drive_velocity_target(self) -> numpy.ndarray[numpy.float32, _Shape[m, 1]]: ...
    def get_force_limit(self):
        return self.force_limit

    def get_friction(self):
        return self.friction

    # def get_global_pose(self) -> sapien.pysapien.Pose: ...
    def get_limits(self):
        return self.limits

    def get_name(self):
        return self.name

    def get_parent_link(self):
        return self.parent_link

    # def get_pose_in_child(self) -> sapien.pysapien.Pose: ...
    # def get_pose_in_parent(self) -> sapien.pysapien.Pose: ...
    def get_stiffness(self) -> float:
        return self.stiffness

    def get_type(self):
        return self.type

    # def set_armature(self, armature: numpy.ndarray[numpy.float32, _Shape[m, 1]]) -> None: ...
    # @before_gpu_init
    def set_drive_properties(
        self,
        stiffness: float,
        damping: float,
        force_limit: float = 3.4028234663852886e38,
        mode: typing.Literal["force", "acceleration"] = "force",
    ):
        for joint in self._objs:
            joint.set_drive_properties(stiffness, damping, force_limit, mode)

    def set_drive_target(self, target: Array) -> None:
        self.drive_target = target

    def set_drive_velocity_target(self, velocity: Array) -> None:
        self.drive_velocity_target = velocity

    def set_friction(self, friction: float):
        self.friction = friction

    def set_limits(self, limits: Array) -> None:
        self.limits = limits

    # def set_name(self, name: str) -> None: ...
    # def set_pose_in_child(self, pose: sapien.pysapien.Pose) -> None: ...
    # def set_pose_in_parent(self, pose: sapien.pysapien.Pose) -> None: ...
    # def set_type(self, type: typing.Literal['fixed', 'revolute', 'revolute_unwrapped', 'prismatic', 'free']) -> None: ...
    # @property
    # def armature(self) -> numpy.ndarray[numpy.float32, _Shape[m, 1]]:
    #     """
    #     :type: numpy.ndarray[numpy.float32, _Shape[m, 1]]
    #     """
    # @armature.setter
    # def armature(self, arg1: numpy.ndarray[numpy.float32, _Shape[m, 1]]) -> None:
    #     pass
    # @property
    # def child_link(self) -> PhysxArticulationLinkComponent:
    #     """
    #     :type: PhysxArticulationLinkComponent
    #     """
    @property
    def damping(self) -> torch.Tensor:
        return torch.tensor([obj.damping for obj in self._objs])

    @property
    def dof(self) -> torch.Tensor:
        return torch.tensor([obj.dof for obj in self._objs])

    @property
    def drive_mode(self) -> List[typing.Literal["force", "acceleration"]]:
        """
        :type: typing.Literal['force', 'acceleration']
        """
        return [obj.drive_mode for obj in self._objs]

    @property
    def drive_target(self) -> torch.Tensor:
        if physx.is_gpu_enabled():
            raise NotImplementedError(
                "Getting drive targets of individual joints is not implemented yet."
            )
        else:
            return torch.from_numpy(self._objs[0].drive_target[None, :])

    @drive_target.setter
    def drive_target(self, arg1: Array) -> None:
        if physx.is_gpu_enabled():
            arg1 = sapien_utils.to_tensor(arg1)
            raise NotImplementedError(
                "Setting drive targets of individual joints is not implemented yet."
            )
        else:
            if arg1.shape == ():
                arg1 = arg1.reshape(
                    1,
                )
            self._objs[0].drive_target = arg1

    @property
    def drive_velocity_target(self) -> torch.Tensor:
        if physx.is_gpu_enabled():
            raise NotImplementedError(
                "Cannot read drive velocity targets at the moment in GPU simulation"
            )
        else:
            return torch.from_numpy(self._objs[0].drive_velocity_target[None, :])

    @drive_velocity_target.setter
    def drive_velocity_target(self, arg1: Array) -> None:
        if physx.is_gpu_enabled():
            arg1 = sapien_utils.to_tensor(arg1)
            raise NotImplementedError(
                "Cannot set drive velocity targets at the moment in GPU simulation"
            )
        else:
            if arg1.shape == ():
                arg1 = arg1.reshape(
                    1,
                )
            self._objs[0].drive_velocity_target = arg1

    @property
    def force_limit(self) -> torch.Tensor:
        return torch.tensor([obj.force_limit for obj in self._objs])

    @property
    def friction(self) -> torch.Tensor:
        return torch.tensor([obj.friction for obj in self._objs])

    @friction.setter
    # @before_gpu_init
    def friction(self, arg1: float) -> None:
        for joint in self._objs:
            joint.friction = arg1

    # @property
    # def global_pose(self) -> sapien.pysapien.Pose:
    #     """
    #     :type: sapien.pysapien.Pose
    #     """
    @property
    def limits(self) -> torch.Tensor:
        # TODO (stao): create a decorator that caches results once gpu sim is initialized for performance
        return torch.from_numpy(np.array([obj.limits for obj in self._objs]))

    @limits.setter
    @before_gpu_init
    def limits(self, arg1: Array) -> None:
        for joint in self._objs:
            joint.limits = arg1

    # @property
    # def name(self) -> str:
    #     """
    #     :type: str
    #     """
    # @name.setter
    # def name(self, arg1: str) -> None:
    #     pass
    # @property
    # def parent_link(self) -> PhysxArticulationLinkComponent:
    #     """
    #     :type: PhysxArticulationLinkComponent
    #     """
    # @property
    # def pose_in_child(self) -> sapien.pysapien.Pose:
    #     """
    #     :type: sapien.pysapien.Pose
    #     """
    # @pose_in_child.setter
    # def pose_in_child(self, arg1: sapien.pysapien.Pose) -> None:
    #     pass
    # @property
    # def pose_in_parent(self) -> sapien.pysapien.Pose:
    #     """
    #     :type: sapien.pysapien.Pose
    #     """
    # @pose_in_parent.setter
    # def pose_in_parent(self, arg1: sapien.pysapien.Pose) -> None:
    #     pass
    @property
    def stiffness(self) -> torch.Tensor:
        return torch.tensor([obj.stiffness for obj in self._objs])

    @property
    def type(
        self,
    ) -> List[
        typing.Literal["fixed", "revolute", "revolute_unwrapped", "prismatic", "free"]
    ]:
        return [obj.type for obj in self._objs]

    @type.setter
    @before_gpu_init
    def type(
        self,
        arg1: typing.Literal[
            "fixed", "revolute", "revolute_unwrapped", "prismatic", "free"
        ],
    ) -> None:
        if physx.is_gpu_enabled():
            return NotImplementedError("Cannot set type of the joint when using GPU")
        else:
            self._objs[0].type = arg1
