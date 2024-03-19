from dataclasses import dataclass

import sapien.physx as physx

from mani_skill.utils.structs import BaseStruct, PhysxRigidBaseComponentStruct, Pose


@dataclass
class PhysxJointComponentStruct(BaseStruct[physx.PhysxJointComponent]):
    # def create(cls, bodies: Sequence[PhysxRigidBodyComponentStruct], parent_bodies: Sequence[PhysxRigidBodyComponentStruct]):
    # TODO
    parent: PhysxRigidBaseComponentStruct
    pose_in_child: Pose
    pose_in_parent: Pose

    # ---------------------------------------------------------------------------- #
    # API from physx.PhysxJointComponent
    # ---------------------------------------------------------------------------- #

    # def get_parent(self) -> PhysxRigidBaseComponent:
    #     ...
    # def get_pose_in_child(self) -> sapien.pysapien.Pose:
    #     ...
    # def get_pose_in_parent(self) -> sapien.pysapien.Pose:
    #     ...
    # def get_relative_pose(self) -> sapien.pysapien.Pose:
    #     ...
    # def set_inv_inertia_scales(self, scale0: float, scale1: float) -> None:
    #     ...
    # def set_inv_mass_scales(self, scale0: float, scale1: float) -> None:
    #     ...
    # def set_parent(self, parent: PhysxRigidBaseComponent) -> None:
    #     ...
    # def set_pose_in_child(self, pose: sapien.pysapien.Pose) -> None:
    #     ...
    # def set_pose_in_parent(self, pose: sapien.pysapien.Pose) -> None:
    #     ...
    # @property
    # def relative_pose(self) -> sapien.pysapien.Pose:
