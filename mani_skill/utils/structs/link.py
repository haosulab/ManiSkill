from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Callable, Dict, List, Union

import sapien
import sapien.physx as physx
import torch
import trimesh

from mani_skill.utils.geometry.trimesh_utils import (
    get_render_shape_meshes,
    merge_meshes,
)
from mani_skill.utils.structs.articulation_joint import ArticulationJoint
from mani_skill.utils.structs.base import PhysxRigidBodyComponentStruct
from mani_skill.utils.structs.pose import Pose, to_sapien_pose, vectorize_pose
from mani_skill.utils.structs.types import Array

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.structs import Articulation


@dataclass
class Link(PhysxRigidBodyComponentStruct[physx.PhysxArticulationLinkComponent]):
    """
    Wrapper around physx.PhysxArticulationLinkComponent objects
    """

    articulation: Articulation = None
    """the articulation that this link is a part of. If this is None, most likely this link object is a view/merged link object in which case
    there is no one articulation that can be referenced easily"""

    name: str = None

    joint: ArticulationJoint = None
    """the joint of which this link is a child of. If this is a view/merged link then this joint is also a view/merged joint"""

    meshes: Dict[str, List[trimesh.Trimesh]] = field(default_factory=dict)
    """
    map from user-defined mesh groups (e.g. "handle" meshes for cabinets) to a list of trimesh.Trimesh objects corresponding to each physx link object managed here
    """

    merged: bool = False
    """whether this link is result of Link.merge or not"""

    def __str__(self):
        return f"<{self.name}: struct of type {self.__class__}; managing {self._num_objs} {self._objs[0].__class__} objects>"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return self.__maniskill_hash__

    @classmethod
    def create(
        cls,
        physx_links: List[physx.PhysxArticulationLinkComponent],
        scene: ManiSkillScene,
        scene_idxs: torch.Tensor,
    ):
        return cls(
            _objs=physx_links,
            scene=scene,
            _scene_idxs=scene_idxs,
            _body_data_name=(
                "cuda_rigid_body_data"
                if isinstance(scene.px, physx.PhysxGpuSystem)
                else None
            ),
            _bodies=physx_links,
        )

    @classmethod
    def merge(cls, links: List["Link"], name: str = None):
        objs = []
        joint_objs = []
        merged_joint_indexes = []
        merged_active_joint_indexes = []
        articulation_objs = []
        is_root = links[0].is_root
        merged_scene_idxs = []
        num_objs_per_actor = links[0]._num_objs
        for link in links:
            objs += link._objs
            assert (
                link.is_root == is_root
            ), "all links given to merge must all be root or all not be root links"
            if not is_root:
                joint_objs += link.joint._objs
                articulation_objs += link.articulation._objs
                merged_scene_idxs.append(link._scene_idxs)
                merged_active_joint_indexes.append(link.joint.active_index)
                merged_joint_indexes.append(link.joint.index)
            assert (
                link._num_objs == num_objs_per_actor
            ), "Each given link must have the same number of managed objects"
        merged_scene_idxs = torch.concat(merged_scene_idxs)
        merged_link = Link.create(
            objs, scene=links[0].scene, scene_idxs=merged_scene_idxs
        )
        if not is_root:
            merged_active_joint_indexes = torch.concat(merged_active_joint_indexes)
            merged_joint_indexes = torch.concat(merged_joint_indexes)
            merged_joint = ArticulationJoint.create(
                joint_objs,
                physx_articulations=articulation_objs,
                scene=links[0].scene,
                scene_idxs=merged_scene_idxs,
                joint_index=merged_joint_indexes,
                active_joint_index=merged_active_joint_indexes,
            )
            merged_link.joint = merged_joint
            merged_joint.child_link = merged_link
        # remove articulation reference as it does not make sense and is only used to instantiate some properties like the physx system
        # TODO (stao): akin to the joint merging above, we can also make a view of the articulations of each link. Is it necessary?
        merged_link.articulation = None
        merged_link.name = name
        merged_link.merged = True
        return merged_link

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
        Generates mesh objects (trimesh.Trimesh) for each managed physx link given a filter and
        saves them to self.meshes[mesh_name] in addition to returning them here.
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

    # TODO (stao): In future can we have some shared nice functions like getting center of mass of desired meshes (e.g. handle mesh?)
    # def get_mesh_center_of_masses(self, mesh_name: str):

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

    def set_collision_group_bit(self, group: int, bit_idx: int, bit: Union[int, bool]):
        """Set's a specific collision group bit for all collision shapes in all parallel actors"""
        bit = int(bit)
        for body in self._bodies:
            for cs in body.get_collision_shapes():
                cg = cs.get_collision_groups()
                cg[group] = (cg[group] & ~(1 << bit_idx)) | (bit << bit_idx)
                cs.set_collision_groups(cg)

    # -------------------------------------------------------------------------- #
    # Functions from sapien.Component
    # -------------------------------------------------------------------------- #
    @property
    def pose(self) -> Pose:
        if physx.is_gpu_enabled():
            # TODO (handle static objects)
            return Pose.create(
                self.px.cuda_rigid_body_data.torch()[self._body_data_index, :7]
            )
        else:
            assert len(self._objs) == 1
            return Pose.create(self._objs[0].entity_pose)

    @pose.setter
    def pose(self, arg1: Union[Pose, sapien.Pose, Array]) -> None:
        if physx.is_gpu_enabled():
            self.px.cuda_rigid_body_data.torch()[
                self._body_data_index[self.scene._reset_mask[self._scene_idxs]], :7
            ] = vectorize_pose(arg1)
        else:
            if isinstance(arg1, sapien.Pose):
                for obj in self._objs:
                    obj.pose = arg1
            else:
                if len(arg1.shape) == 2:
                    for i, obj in enumerate(self._objs):
                        obj.pose = arg1[i].sp
                else:
                    arg1 = to_sapien_pose(arg1)
                    for i, obj in enumerate(self._objs):
                        obj.pose = arg1

    def set_pose(self, arg1: Union[Pose, sapien.Pose]) -> None:
        self.pose = arg1

    # -------------------------------------------------------------------------- #
    # Functions from physx.PhysxArticulationLinkComponent
    # -------------------------------------------------------------------------- #
    def get_articulation(self):
        return self.articulation

    # def get_children(self) -> list[PhysxArticulationLinkComponent]: ...
    # def get_gpu_pose_index(self) -> int: ...
    def get_index(self):
        return self.index

    def get_joint(self) -> ArticulationJoint:
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
    @cached_property
    def index(self) -> torch.Tensor:
        """The indexes of the managed link objects in their respective articulations. NOTE that these do not correspond with position in the qpos and qvel of articulations. For that index use index_q"""
        return torch.tensor(
            [obj.index for obj in self._objs], dtype=torch.int, device=self.device
        )

    @cached_property
    def is_root(self) -> torch.Tensor:
        return torch.tensor(
            [obj.is_root for obj in self._objs], dtype=torch.bool, device=self.device
        )

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
