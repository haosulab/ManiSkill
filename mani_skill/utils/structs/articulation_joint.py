from __future__ import annotations

import typing
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import sapien.physx as physx
import torch

from mani_skill.utils import common
from mani_skill.utils.structs.base import BaseStruct
from mani_skill.utils.structs.decorators import before_gpu_init
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.structs.articulation import Articulation
    from mani_skill.utils.structs.link import Link


@dataclass
class ArticulationJoint(BaseStruct[physx.PhysxArticulationJoint]):
    """
    Wrapper around physx.PhysxArticulationJoint objects

    At the moment, all of the same joints across all sub scenes are restricted to having the same properties as each other
    """

    index: torch.Tensor
    """index of this joint among all joints"""
    active_index: torch.Tensor
    """index of this joint amongst the active joints"""

    articulation: Optional[Articulation] = None
    child_link: Optional[Link] = None
    parent_link: Optional[Link] = None
    name: str = None

    _physx_articulations: List[physx.PhysxArticulation] = None

    def __str__(self):
        return f"<{self.name}: struct of type {self.__class__}; managing {self._num_objs} {self._objs[0].__class__} objects>"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return self.__maniskill_hash__

    @classmethod
    def create(
        cls,
        physx_joints: List[physx.PhysxArticulationJoint],
        physx_articulations: List[physx.PhysxArticulation],
        scene: ManiSkillScene,
        scene_idxs: torch.Tensor,
        joint_index: torch.Tensor,
        active_joint_index: torch.Tensor = None,
    ):
        """Creates an object for managing articulation joints in articulations

        Note that the properties articulation, child_link, parent_link are by default None
        as they might not make sense in GPU sim and must be set by user
        """
        return cls(
            index=joint_index,
            active_index=active_joint_index,
            _objs=physx_joints,
            _physx_articulations=physx_articulations,
            scene=scene,
            _scene_idxs=scene_idxs,
        )

    # -------------------------------------------------------------------------- #
    # Additional useful functions not in SAPIEN original API
    # -------------------------------------------------------------------------- #
    @cached_property
    def _data_index(self):
        return torch.tensor(
            [
                px_articulation.gpu_index
                for px_articulation in self._physx_articulations
            ],
            device=self.device,
            dtype=torch.int32,
        )

    @property
    def qpos(self):
        """
        The qpos of this joint in the articulation
        """
        assert (
            self.active_index is not None
        ), "Inactive joints do not have qpos/qvel values"
        if self.scene.gpu_sim_enabled:
            return self.px.cuda_articulation_qpos.torch()[
                self._data_index, self.active_index
            ]
        else:
            return torch.tensor([self._physx_articulations[0].qpos[self.active_index]])

    @property
    def qvel(self):
        """
        The qvel of this joint in the articulation
        """
        assert (
            self.active_index is not None
        ), "Inactive joints do not have qpos/qvel values"
        if self.scene.gpu_sim_enabled:
            return self.px.cuda_articulation_qvel.torch()[
                self._data_index, self.active_index
            ]
        else:
            return torch.tensor([self._physx_articulations[0].qvel[self.active_index]])

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

    def get_global_pose(self):
        return self.global_pose

    def get_limits(self):
        return self.limits

    def get_name(self):
        return self.name

    def get_parent_link(self):
        return self.parent_link

    def get_pose_in_child(self):
        return self.pose_in_child

    def get_pose_in_parent(self):
        return self.pose_in_parent

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
        if self.scene.gpu_sim_enabled:
            raise NotImplementedError(
                "Getting drive targets of individual joints is not implemented yet."
            )
        else:
            return torch.from_numpy(self._objs[0].drive_target[None, :])

    @drive_target.setter
    def drive_target(self, arg1: Array) -> None:
        arg1 = common.to_tensor(arg1, device=self.device)
        if self.scene.gpu_sim_enabled:
            raise NotImplementedError(
                "Setting drive targets of individual joints is not implemented yet."
            )
        else:
            if arg1.shape == ():
                arg1 = arg1.reshape(
                    1,
                )
            self._objs[0].drive_target = arg1.numpy()

    @property
    def drive_velocity_target(self) -> torch.Tensor:
        if self.scene.gpu_sim_enabled:
            raise NotImplementedError(
                "Cannot read drive velocity targets at the moment in GPU simulation"
            )
        else:
            return torch.from_numpy(self._objs[0].drive_velocity_target[None, :])

    @drive_velocity_target.setter
    def drive_velocity_target(self, arg1: Array) -> None:
        arg1 = common.to_tensor(arg1, device=self.device)
        if self.scene.gpu_sim_enabled:
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
    # TODO (stao): can we set this after gpu is initialized?
    def friction(self, arg1: float) -> None:
        for joint in self._objs:
            joint.friction = arg1

    @property
    def global_pose(self) -> Pose:
        return self.pose_in_child * self.child_link.pose

    @property
    def limits(self) -> torch.Tensor:
        # TODO (stao): create a decorator that caches results once gpu sim is initialized for performance
        return common.to_tensor(
            np.array([obj.limits[0] for obj in self._objs]), device=self.device
        )

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
    @cached_property
    def pose_in_child(self):
        raw_poses = np.stack(
            [np.concatenate([x.pose_in_child.p, x.pose_in_child.q]) for x in self._objs]
        )
        return Pose.create(common.to_tensor(raw_poses), device=self.device)

    # @pose_in_child.setter
    # def pose_in_child(self, arg1: sapien.pysapien.Pose) -> None:
    #     pass
    @cached_property
    def pose_in_parent(self):
        raw_poses = np.stack(
            [
                np.concatenate([x.pose_in_parent.p, x.pose_in_parent.q])
                for x in self._objs
            ]
        )
        return Pose.create(common.to_tensor(raw_poses, device=self.device))

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
        if self.scene.gpu_sim_enabled:
            return NotImplementedError("Cannot set type of the joint when using GPU")
        else:
            self._objs[0].type = arg1
