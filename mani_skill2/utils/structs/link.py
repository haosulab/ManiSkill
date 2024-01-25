from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Union

import sapien
import sapien.physx as physx

from mani_skill2.utils.structs.base import BaseStruct, PhysxRigidBodyComponentStruct

if TYPE_CHECKING:
    from mani_skill2.utils.structs.articulation import Articulation

from mani_skill2.utils.structs.pose import Pose, to_sapien_pose, vectorize_pose
from mani_skill2.utils.structs.types import Array


@dataclass
class Link(
    PhysxRigidBodyComponentStruct, BaseStruct[physx.PhysxArticulationLinkComponent]
):
    """
    Wrapper around physx.PhysxArticulationLinkComponent objects
    """

    articulation: Articulation

    name: str = None

    @classmethod
    def create(
        cls,
        physx_links: List[physx.PhysxArticulationLinkComponent],
        articulation: Articulation,
    ):
        shared_name = "_".join(physx_links[0].name.split("_")[1:])
        return cls(
            articulation=articulation,
            _objs=physx_links,
            _scene=articulation._scene,
            _scene_mask=articulation._scene_mask,
            name=shared_name,
            _body_data_name="cuda_rigid_body_data"
            if isinstance(articulation.px, physx.PhysxGpuSystem)
            else None,
            _bodies=physx_links,
            _body_data_index=None,
        )

    # -------------------------------------------------------------------------- #
    # Functions from sapien.Component
    # -------------------------------------------------------------------------- #
    @property
    def pose(self) -> Pose:
        if physx.is_gpu_enabled():
            # TODO (handle static objects)
            return Pose.create(self.px.cuda_rigid_body_data[self._body_data_index, :7])
        else:
            assert len(self._objs) == 1
            return Pose.create(self._objs[0].pose)

    @pose.setter
    def pose(self, arg1: Union[Pose, sapien.Pose]) -> None:
        if physx.is_gpu_enabled():
            self.px.cuda_rigid_body_data[self._body_data_index, :7] = vectorize_pose(
                arg1
            )
        else:
            self._objs[0].pose = to_sapien_pose(arg1)

    def set_pose(self, arg1: Union[Pose, sapien.Pose]) -> None:
        self.pose = arg1

    # -------------------------------------------------------------------------- #
    # Functions from physx.PhysxArticulationLinkComponent
    # -------------------------------------------------------------------------- #
    def get_articulation(self) -> physx.PhysxArticulation:
        return self.articulation

    # def get_children(self) -> list[PhysxArticulationLinkComponent]: ...
    # def get_gpu_pose_index(self) -> int: ...
    def get_index(self) -> int:
        return self.index

    def get_joint(self) -> physx.PhysxArticulationJoint:
        return self.joint

    # def get_parent(self) -> PhysxArticulationLinkComponent: ...
    # def put_to_sleep(self) -> None: ...
    # def set_parent(self, parent: PhysxArticulationLinkComponent) -> None: ...
    # def wake_up(self) -> None: ...

    # @property
    # def children(self) -> list[PhysxArticulationLinkComponent]:
    #     """
    #     :type: list[PhysxArticulationLinkComponent]
    #     """
    # @property
    # def gpu_pose_index(self) -> int:
    #     """
    #     :type: int
    #     """
    @property
    def index(self) -> int:
        """
        :type: int
        """
        return self._objs[0].get_index()

    @property
    def is_root(self) -> bool:
        """
        :type: bool
        """
        return self._objs[0].is_root

    @property
    def joint(self) -> physx.PhysxArticulationJoint:
        """
        :type: PhysxArticulationJoint
        """
        # TODO (stao): make joint struct?
        return self.joint

    # @property
    # def parent(self) -> PhysxArticulationLinkComponent:
    #     """
    #     :type: PhysxArticulationLinkComponent
    #     """
    # @property
    # def sleeping(self) -> bool:
    #     """
    #     :type: bool
    #     """

    # @property
    # def entity(self):
    #     return self._links[0]

    def get_name(self):
        return self.name
