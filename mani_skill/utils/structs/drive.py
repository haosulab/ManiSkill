from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Sequence, Union

import sapien
import sapien.physx as physx
import torch

from mani_skill.utils.structs import (
    Actor,
    Link,
    PhysxJointComponentStruct,
    PhysxRigidBaseComponentStruct,
    Pose,
)
from mani_skill.utils.structs.decorators import before_gpu_init

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene


@dataclass
class Drive(PhysxJointComponentStruct[physx.PhysxDriveComponent]):
    # drive_target: Pose # TODO (stao): what is this?

    def __hash__(self):
        return self.__maniskill_hash__

    @classmethod
    def create_from_entities(
        cls,
        scene: ManiSkillScene,
        bodies0: Sequence[Union[sapien.Entity, physx.PhysxRigidBaseComponent]] = None,
        pose0: Union[sapien.Pose, Pose] = None,
        bodies1: Sequence[Union[sapien.Entity, physx.PhysxRigidBaseComponent]] = None,
        pose1: Union[sapien.Pose, Pose] = None,
        scene_idxs: torch.Tensor = None,
    ):
        physx_drives: List[physx.PhysxDriveComponent] = []
        assert bodies1 is not None
        if bodies0 is None:
            bodies0 = [None] * len(bodies1)
        if scene_idxs is None:
            scene_idxs = torch.arange(0, scene.num_envs)
        assert len(scene_idxs) == len(bodies0)
        pose1 = Pose.create(pose1)
        pose0 = Pose.create(pose0)
        for i in scene_idxs:
            sub_scene = scene.sub_scenes[i]
            physx_drives.append(
                sub_scene.create_drive(bodies0[i], pose0[0].sp, bodies1[i], pose1[0].sp)
            )
        # NOTE (stao): SAPIEN structure might have some inconsistency? For drives there is no such thing as bodies / being able to compute
        # aabbs
        return cls(
            _objs=physx_drives,
            _scene_idxs=scene_idxs,
            pose_in_child=pose1,
            pose_in_parent=pose0,
            scene=scene,
        )

    @staticmethod
    def create_from_actors_or_links(
        scene: ManiSkillScene,
        entities0: Union[Actor, Link] = None,
        pose0: Union[sapien.Pose, Pose] = None,
        entities1: Union[Actor, Link] = None,
        pose1: Union[sapien.Pose, Pose] = None,
        scene_idxs: torch.Tensor = None,
    ) -> "Drive":
        """create a batched drive between two Actors/Links"""
        objs0 = entities0._objs
        objs1 = entities1._objs
        if isinstance(entities0, Link):
            objs0 = [x.entity for x in objs0]
        if isinstance(entities1, Link):
            objs1 = [x.entity for x in objs1]

        return Drive.create_from_entities(scene, objs0, pose0, objs1, pose1, scene_idxs)

    # def __init__(self, body: PhysxRigidBodyComponent) -> None:
    #     ...
    # def get_drive_property_slerp(self) -> tuple[float, float, float, typing.Literal['force', 'acceleration']]:
    #     ...
    # def get_drive_property_swing(self) -> tuple[float, float, float, typing.Literal['force', 'acceleration']]:
    #     ...
    # def get_drive_property_twist(self) -> tuple[float, float, float, typing.Literal['force', 'acceleration']]:
    #     ...
    # def get_drive_property_x(self) -> tuple[float, float, float, typing.Literal['force', 'acceleration']]:
    #     ...
    # def get_drive_property_y(self) -> tuple[float, float, float, typing.Literal['force', 'acceleration']]:
    #     ...
    # def get_drive_property_z(self) -> tuple[float, float, float, typing.Literal['force', 'acceleration']]:
    #     ...
    # def get_drive_target(self) -> sapien.pysapien.Pose:
    #     ...
    # def get_drive_velocity_target(self) -> tuple[numpy.ndarray[typing.Literal[3], numpy.dtype[numpy.float32]], numpy.ndarray[typing.Literal[3], numpy.dtype[numpy.float32]]]:
    #     ...
    # def get_limit_cone(self) -> tuple[float, float, float, float]:
    #     ...
    # def get_limit_pyramid(self) -> tuple[float, float, float, float, float, float]:
    #     ...
    # def get_limit_twist(self) -> tuple[float, float, float, float]:
    #     ...
    # def get_limit_x(self) -> tuple[float, float, float, float]:
    #     ...
    # def get_limit_y(self) -> tuple[float, float, float, float]:
    #     ...
    # def get_limit_z(self) -> tuple[float, float, float, float]:
    #     ...
    # def set_drive_property_slerp(self, stiffness: float, damping: float, force_limit: float = 3.4028234663852886e+38, mode: typing.Literal['force', 'acceleration'] = 'force') -> None:
    #     ...
    # def set_drive_property_swing(self, stiffness: float, damping: float, force_limit: float = 3.4028234663852886e+38, mode: typing.Literal['force', 'acceleration'] = 'force') -> None:
    #     ...
    # def set_drive_property_twist(self, stiffness: float, damping: float, force_limit: float = 3.4028234663852886e+38, mode: typing.Literal['force', 'acceleration'] = 'force') -> None:
    #     ...

    # TODO (stao): permit providing batched values
    @before_gpu_init
    def set_drive_property_x(
        self,
        stiffness: float,
        damping: float,
        force_limit: float = 3.4028234663852886e38,
        mode: typing.Literal["force", "acceleration"] = "force",
    ) -> None:
        [
            x.set_drive_property_x(stiffness, damping, force_limit, mode)
            for x in self._objs
        ]

    @before_gpu_init
    def set_drive_property_y(
        self,
        stiffness: float,
        damping: float,
        force_limit: float = 3.4028234663852886e38,
        mode: typing.Literal["force", "acceleration"] = "force",
    ) -> None:
        [
            x.set_drive_property_y(stiffness, damping, force_limit, mode)
            for x in self._objs
        ]

    @before_gpu_init
    def set_drive_property_z(
        self,
        stiffness: float,
        damping: float,
        force_limit: float = 3.4028234663852886e38,
        mode: typing.Literal["force", "acceleration"] = "force",
    ) -> None:
        [
            x.set_drive_property_z(stiffness, damping, force_limit, mode)
            for x in self._objs
        ]

    # def set_drive_target(self, target: sapien.pysapien.Pose) -> None:
    #     ...
    # def set_drive_velocity_target(self, linear: numpy.ndarray[typing.Literal[3], numpy.dtype[numpy.float32]] | list[float] | tuple, angular: numpy.ndarray[typing.Literal[3], numpy.dtype[numpy.float32]] | list[float] | tuple) -> None:
    #     ...
    # def set_limit_cone(self, angle_y: float, angle_z: float, stiffness: float = 0.0, damping: float = 0.0) -> None:
    #     ...
    # def set_limit_pyramid(self, low_y: float, high_y: float, low_z: float, high_z: float, stiffness: float = 0.0, damping: float = 0.0) -> None:
    #     ...
    # def set_limit_twist(self, low: float, high: float, stiffness: float = 0.0, damping: float = 0.0) -> None:
    #     ...
    @before_gpu_init
    def set_limit_x(
        self, low: float, high: float, stiffness: float = 0.0, damping: float = 0.0
    ) -> None:
        [x.set_limit_x(low, high, stiffness, damping) for x in self._objs]

    @before_gpu_init
    def set_limit_y(
        self, low: float, high: float, stiffness: float = 0.0, damping: float = 0.0
    ) -> None:
        [x.set_limit_y(low, high, stiffness, damping) for x in self._objs]

    @before_gpu_init
    def set_limit_z(
        self, low: float, high: float, stiffness: float = 0.0, damping: float = 0.0
    ) -> None:
        [x.set_limit_z(low, high, stiffness, damping) for x in self._objs]
