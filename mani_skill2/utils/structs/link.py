from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Union

import sapien
import sapien.physx as physx
import trimesh

from mani_skill2.utils.geometry.trimesh_utils import (
    get_render_shape_meshes,
    merge_meshes,
)
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

    articulation: Articulation = None

    name: str = None

    meshes: Dict[str, List[trimesh.Trimesh]] = field(default_factory=dict)
    """
    map from user-defined mesh groups (e.g. "handle" meshes for cabinets) to a list of trimesh.Trimesh objects corresponding to each physx link object managed here
    """

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
        )

    # -------------------------------------------------------------------------- #
    # Additional useful functions not in SAPIEN original API
    # -------------------------------------------------------------------------- #
    @property
    def render_shapes(self):
        """
        Returns each managed link objects render shape list (a list of lists)
        """
        all_render_shapes: List[List[sapien.render.RenderShape]] = []
        for obj in self._objs:
            all_render_shapes.append(
                obj.entity.find_component_by_type(
                    sapien.render.RenderBodyComponent
                ).render_shapes
            )
        return all_render_shapes

    def generate_mesh(
        self,
        filter: Callable[
            [physx.PhysxArticulationLinkComponent, sapien.render.RenderShape], bool
        ],
        mesh_name: str,
    ):
        """
        Generates mesh objects (trimesh.Trimesh) for each managed physx link and saves them to self.meshes[mesh_name] in addition to returning them here.
        This does this by
        """
        # TODO (stao): should we create a Mesh wrapper? that is basically trimesh.Trimesh but all batched.
        if mesh_name in self.meshes:
            return self.meshes[mesh_name]
        merged_meshes = []
        for link, link_render_shapes in zip(self._objs, self.render_shapes):
            meshes = []
            for render_shape in link_render_shapes:
                if filter(link, render_shape):
                    meshes.extend(get_render_shape_meshes(render_shape))
            merged_meshes.append(merge_meshes(meshes))
        self.meshes[mesh_name] = merged_meshes
        return merged_meshes

    def bbox(
        self,
        filter: Callable[
            [physx.PhysxArticulationLinkComponent, sapien.render.RenderShape], bool
        ],
    ) -> List[trimesh.primitives.Box]:
        # First we need to pre-compute the bounding box of the link at 0. This will be slow the first time
        bboxes = []
        for link, link_render_shapes in zip(self._objs, self.render_shapes):
            meshes = []
            for render_shape in link_render_shapes:
                if filter(link, render_shape):
                    meshes.extend(get_render_shape_meshes(render_shape))
            merged_mesh = merge_meshes(meshes)
            bboxes.append(merged_mesh.bounding_box)
        return bboxes

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

    # def get_joint(self) -> physx.PhysxArticulationJoint:
    #     return self.joint

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

    # @property
    # def joint(self) -> physx.PhysxArticulationJoint:
    #     """
    #     :type: PhysxArticulationJoint
    #     """
    #     return self.joint

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
