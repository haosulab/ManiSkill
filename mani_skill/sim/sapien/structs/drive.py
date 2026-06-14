from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence, Union

import sapien
import sapien.physx as physx
import torch

from mani_skill.sim.sapien.structs.actor import SapienActor
from mani_skill.sim.sapien.structs.base import PhysxJointComponentStruct
from mani_skill.sim.sapien.structs.decorators import before_gpu_init
from mani_skill.sim.sapien.structs.link import SapienLink
from mani_skill.utils.structs.pose import Pose

if TYPE_CHECKING:
    from mani_skill.sim.sapien.sim import SapienSim


@dataclass
class Drive(PhysxJointComponentStruct[physx.PhysxDriveComponent]):
    # drive_target: Pose # TODO (stao): what is this?

    def __hash__(self):
        return self.__maniskill_hash__

    @classmethod
    def create_from_entities(
        cls,
        sim: SapienSim,
        bodies0: Sequence[Union[sapien.Entity, physx.PhysxRigidBaseComponent]] = None,
        pose0: Union[sapien.Pose, Pose] = None,
        bodies1: Sequence[Union[sapien.Entity, physx.PhysxRigidBaseComponent]] = None,
        pose1: Union[sapien.Pose, Pose] = None,
        scene_idxs: torch.Tensor = None,
    ):
        """Create a batched drive from raw SAPIEN entities or rigid body components.

        Args:
            sim: The SAPIEN simulation backend to create drives in.
            bodies0: Parent bodies for each sub-scene. ``None`` entries create world drives.
            pose0: Pose of the drive frame in the parent body.
            bodies1: Child bodies for each sub-scene.
            pose1: Pose of the drive frame in the child body.
            scene_idxs: Indexes of the sub-scenes to create drives in.

        Returns:
            A batched Drive struct managing one drive per selected sub-scene.
        """
        physx_drives: list[physx.PhysxDriveComponent] = []
        assert bodies1 is not None
        if bodies0 is None:
            bodies0 = [None] * len(bodies1)
        if scene_idxs is None:
            scene_idxs = torch.arange(0, sim.num_envs)
        assert len(scene_idxs) == len(bodies0)
        pose1 = Pose.create(pose1)
        pose0 = Pose.create(pose0)
        for i in scene_idxs:
            sub_scene = sim.sub_scenes[i]
            physx_drives.append(
                sub_scene.create_drive(bodies0[i], pose0[0].sp, bodies1[i], pose1[0].sp)
            )
        # NOTE (stao): SAPIEN structure might have some inconsistency? For drives there is no
        # such thing as bodies / being able to compute aabbs
        return cls(
            _objs=physx_drives,
            _scene_idxs=scene_idxs,
            pose_in_child=pose1,
            pose_in_parent=pose0,
            sim=sim,
        )

    @staticmethod
    def create_from_actors_or_links(
        sim: SapienSim,
        entities0: Union[SapienActor, SapienLink] = None,
        pose0: Union[sapien.Pose, Pose] = None,
        entities1: Union[SapienActor, SapienLink] = None,
        pose1: Union[sapien.Pose, Pose] = None,
        scene_idxs: torch.Tensor = None,
    ) -> "Drive":
        """Create a batched drive between two Actors/Links.

        Args:
            sim: The SAPIEN simulation backend to create drives in.
            entities0: Parent actor or link for the drive.
            pose0: Pose of the drive frame in the parent body.
            entities1: Child actor or link for the drive.
            pose1: Pose of the drive frame in the child body.
            scene_idxs: Indexes of the sub-scenes to create drives in.

        Returns:
            A batched Drive struct managing one drive per selected sub-scene.
        """
        objs0 = entities0._objs
        objs1 = entities1._objs
        if isinstance(entities0, SapienLink):
            objs0 = [x.entity for x in objs0]
        if isinstance(entities1, SapienLink):
            objs1 = [x.entity for x in objs1]

        return Drive.create_from_entities(sim, objs0, pose0, objs1, pose1, scene_idxs)

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
