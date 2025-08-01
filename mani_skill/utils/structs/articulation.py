from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import sapien
import sapien.physx as physx
import torch
import trimesh

from mani_skill.utils import common, sapien_utils
from mani_skill.utils.geometry.trimesh_utils import (
    get_component_meshes,
    get_render_shape_meshes,
    merge_meshes,
)
from mani_skill.utils.structs import ArticulationJoint, BaseStruct, Link, Pose
from mani_skill.utils.structs.types import Array

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene


@dataclass
class Articulation(BaseStruct[physx.PhysxArticulation]):
    """
    Wrapper around physx.PhysxArticulation objects
    """

    links: List[Link]
    """List of Link objects"""
    links_map: Dict[str, Link]
    """Maps link name to the Link object"""
    root: Link
    """The root Link object"""
    joints: List[ArticulationJoint]
    """List of Joint objects"""
    joints_map: Dict[str, ArticulationJoint]
    """Maps joint name to the Joint object"""
    active_joints: List[ArticulationJoint]
    """List of active Joint objects, referencing elements in self.joints"""
    active_joints_map: Dict[str, ArticulationJoint]
    """Maps active joint name to the Joint object, referencing elements in self.joints"""

    name: str = None
    """Name of this articulation"""
    initial_pose: Pose = None
    """The initial pose of this articulation"""

    merged: bool = False
    """
    whether or not this articulation object is a merged articulation where it is managing many articulations with different dofs.

    There are a number of caveats when it comes to merged articulations. While merging articulations means you can easily fetch padded
    qpos, qvel, etc. type data, a number of attributes and functions will make little sense and you should avoid using them unless you
    are an advanced user. In particular, the list of Links, Joints, their corresponding maps, net contact forces of multiple links, no
    longer make "sense"
    """

    _cached_joint_target_indices: Dict[int, torch.Tensor] = field(default_factory=dict)
    """Map from a set of joints of this articulation and the indexing torch tensor to use for setting drive targets in GPU sims."""

    _net_contact_force_queries: Dict[
        Tuple, physx.PhysxGpuContactBodyImpulseQuery
    ] = field(default_factory=dict)
    """Maps a tuple of link names to pre-saved net contact force queries"""

    def __str__(self):
        return f"<{self.name}: struct of type {self.__class__}; managing {self._num_objs} {self._objs[0].__class__} objects>"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return self.__maniskill_hash__

    @classmethod
    def create_from_physx_articulations(
        cls,
        physx_articulations: List[physx.PhysxArticulation],
        scene: ManiSkillScene,
        scene_idxs: torch.Tensor,
        _merged: bool = False,
        _process_links: bool = True,
    ) -> Articulation:
        """
        Create a managed articulation object given a list of physx articulations. Note that this function requires all given articulations
        to be the same articulations. To create an object to manage different articulations use the .merge function.
        """
        shared_name = "_".join(physx_articulations[0].name.split("_")[1:])

        # NOTE (stao): This is a little bit of a non-standard way to use @classmethod style creation functions for dataclasses
        # however generating the Joint and Link objects relies on a reference to the articulation properties so this is done as a
        # convenience
        self = cls(
            _objs=physx_articulations,
            scene=scene,
            _scene_idxs=scene_idxs,
            links=None,
            links_map=None,
            root=None,
            joints=None,
            joints_map=None,
            active_joints=None,
            active_joints_map=None,
            name=shared_name,
            merged=_merged,
        )
        # create link and joint structs
        num_links = max([len(x.links) for x in physx_articulations])
        all_links_objs: List[List[physx.PhysxArticulationLinkComponent]] = [
            [] for _ in range(num_links)
        ]
        num_joints = max([len(x.joints) for x in physx_articulations])
        all_joint_objs: List[List[physx.PhysxArticulationJoint]] = [
            [] for _ in range(num_joints)
        ]

        links_map: Dict[str, Link] = dict()
        for articulation in physx_articulations:
            if _process_links:
                assert num_links == len(articulation.links) and num_joints == len(
                    articulation.joints
                ), "Gave different physx articulations. Articulation object created via create_from_physx_articulations can only \
                    manage the same articulations, not different ones. Use merge instead if you want to manage different articulations"
            for i, link in enumerate(articulation.links):
                all_links_objs[i].append(link)
            for i, joint in enumerate(articulation.joints):
                all_joint_objs[i].append(joint)
        wrapped_links: List[Link] = []

        root = None
        for links in all_links_objs:
            wrapped_link = Link.create(links, scene, scene_idxs)
            wrapped_link.name = "_".join(
                links[0].name.replace(self.name, "", 1).split("_")[1:]
            )
            wrapped_link.articulation = self
            links_map[wrapped_link.name] = wrapped_link
            wrapped_links.append(wrapped_link)
            assert wrapped_link.is_root.any() == wrapped_link.is_root.all()
            if wrapped_link.is_root.any():
                root = wrapped_link
                if not _process_links:
                    # if not processing links we break early so that we only processed the root link which is always shared between articulations
                    break
        assert root is not None, "root link was not found"
        self.root = root

        if _process_links:
            self.links = wrapped_links
            self.links_map = links_map

            # create Joint objects and figure out active joint references
            all_active_joint_names = [
                x.name for x in physx_articulations[0].get_active_joints()
            ]
            all_joint_names = [x.name for x in physx_articulations[0].joints]
            active_joint_indices = [
                all_joint_names.index(x) for x in all_active_joint_names
            ]

            joints_map = dict()
            wrapped_joints: List[ArticulationJoint] = []
            for joint_index, joints in enumerate(all_joint_objs):
                try:
                    active_joint_index = all_active_joint_names.index(
                        all_joint_names[joint_index]
                    )
                except:
                    active_joint_index = None
                wrapped_joint = ArticulationJoint.create(
                    physx_joints=joints,
                    physx_articulations=physx_articulations,
                    scene=scene,
                    scene_idxs=scene_idxs,
                    joint_index=torch.zeros(
                        len(joints), dtype=torch.int32, device=self.device
                    )
                    + joint_index,
                    active_joint_index=(
                        torch.zeros(len(joints), dtype=torch.int32, device=self.device)
                        + active_joint_index
                        if active_joint_index is not None
                        else None
                    ),
                )
                wrapped_joint.name = "_".join(
                    joints[0].name.replace(self.name, "", 1).split("_")[1:]
                )
                wrapped_joint.articulation = self
                joints_map[wrapped_joint.name] = wrapped_joint
                wrapped_joints.append(wrapped_joint)
            self.joints = wrapped_joints
            self.joints_map = joints_map
            self.active_joints = [wrapped_joints[i] for i in active_joint_indices]
            self.active_joints_map = {joint.name: joint for joint in self.active_joints}

            # add references of joints to links and links to joints
            for joint in self.joints:
                if joint._objs[0].child_link is not None:
                    joint.child_link = self.links_map[
                        "_".join(
                            joint._objs[0]
                            .child_link.name.replace(self.name, "", 1)
                            .split("_")[1:]
                        )
                    ]
                    joint.child_link.joint = joint
                if joint._objs[0].parent_link is not None:
                    joint.parent_link = self.links_map[
                        "_".join(
                            joint._objs[0]
                            .parent_link.name.replace(self.name, "", 1)
                            .split("_")[1:]
                        )
                    ]
        return self

    @classmethod
    def merge(
        cls,
        articulations: List["Articulation"],
        name: str = None,
        merge_links: bool = False,
    ):
        """
        Merge a list of articulations into a single articulation for easy access of data across multiple possibly different articulations.

        Args:
            articulations: A list of articulations objects to merge.
            name: The name of the merged articulation.
            merge_links: Whether to merge the links of the articulations. This is by default False as often times you merge articulations
                that have different number of links. Set this true if you want to try and merge articulations that have the same number of links.
        """
        objs = []
        scene = articulations[0].scene
        merged_scene_idxs = []
        num_objs_per_actor = articulations[0]._num_objs
        for articulation in articulations:
            objs += articulation._objs
            merged_scene_idxs.append(articulation._scene_idxs)
            assert (
                articulation._num_objs == num_objs_per_actor
            ), "Each given articulation must have the same number of managed objects"
        merged_scene_idxs = torch.concat(merged_scene_idxs)
        merged_articulation = Articulation.create_from_physx_articulations(
            objs, scene, merged_scene_idxs, _merged=True, _process_links=merge_links
        )
        merged_articulation.name = name
        scene.articulation_views[merged_articulation.name] = merged_articulation
        return merged_articulation

    # -------------------------------------------------------------------------- #
    # Additional useful functions not in SAPIEN original API
    # -------------------------------------------------------------------------- #

    @cached_property
    def _data_index(self):
        """
        Returns a tensor of the indices of the articulation in the GPU simulation for the physx_cuda backend.
        """
        return torch.tensor(
            [px_articulation.gpu_index for px_articulation in self._objs],
            device=self.device,
            dtype=torch.int32,
        )

    @cached_property
    def fixed_root_link(self):
        """
        Returns a boolean tensor of whether the root link is fixed for each parallel articulation
        """
        return torch.tensor(
            [x.links[0].entity.components[0].joint.type == "fixed" for x in self._objs],
            device=self.device,
            dtype=torch.bool,
        )

    def get_state(self):
        pose = self.root.pose
        vel = self.root.get_linear_velocity()  # [N, 3]
        ang_vel = self.root.get_angular_velocity()  # [N, 3]
        qpos = self.get_qpos()
        qvel = self.get_qvel()
        return torch.hstack([pose.p, pose.q, vel, ang_vel, qpos, qvel])

    def set_state(self, state: Array, env_idx: torch.Tensor = None):
        if self.scene.gpu_sim_enabled:
            if env_idx is not None:
                prev_reset_mask = self.scene._reset_mask.clone()
                # safe guard against setting the wrong states
                self.scene._reset_mask[:] = False
                self.scene._reset_mask[env_idx] = True
            state = common.to_tensor(state, device=self.device)
            self.set_root_pose(Pose.create(state[:, :7]))
            self.set_root_linear_velocity(state[:, 7:10])
            self.set_root_angular_velocity(state[:, 10:13])
            self.set_qpos(state[:, 13 : 13 + self.max_dof])
            self.set_qvel(state[:, 13 + self.max_dof : 13 + self.max_dof * 2])
            if env_idx is not None:
                self.scene._reset_mask = prev_reset_mask
        else:
            state = common.to_numpy(state[0])
            self.set_root_pose(sapien.Pose(state[0:3], state[3:7]))
            self.set_root_linear_velocity(state[7:10])
            self.set_root_angular_velocity(state[10:13])
            qpos, qvel = np.split(state[13 : 13 + self.max_dof * 2], 2)
            self.set_qpos(qpos)
            self.set_qvel(qvel)

    @cached_property
    def max_dof(self) -> int:
        """the max DOF out of all managed objects. This is used to padd attributes like qpos"""
        return max([obj.dof for obj in self._objs])

    # currently does not work in SAPIEN. Must set collision groups prior to building Link
    # def disable_self_collisions(self):
    #     """Disable all self collisions between links. Note that 1 << 31 is reserved in the entirety of ManiSkill for link self collisions"""
    #     for link in self.links:
    #         for obj in link._objs:
    #             for s in obj.get_collision_shapes():
    #                 g0, g1, g2, g3 = s.get_collision_groups()
    #                 s.set_collision_groups([g0, g1, g2 | (1 << 29), g3])

    def get_first_collision_mesh(
        self, to_world_frame: bool = True
    ) -> Union[trimesh.Trimesh, None]:
        """
        Returns the collision mesh of the first managed articulation object. Note results of this are not cached or optimized at the moment
        so this function can be slow if called too often. Some articulations have no collision meshes, in which case this function returns None

        Args:
            to_world_frame (bool): Whether to transform the collision mesh pose to the world frame
        """
        mesh = self.get_collision_meshes(to_world_frame=to_world_frame, first_only=True)
        if isinstance(mesh, trimesh.Trimesh):
            return mesh
        return None

    def get_collision_meshes(
        self, to_world_frame: bool = True, first_only: bool = False
    ) -> Union[List[trimesh.Trimesh], trimesh.Trimesh]:
        """
        Returns the collision mesh of each managed articulation object. Note results of this are not cached or optimized at the moment
        so this function can be slow if called too often

        Args:
            to_world_frame (bool): Whether to transform the collision mesh pose to the world frame
            first_only (bool): Whether to return the collision mesh of just the first articulation managed by this object. If True,
                this also returns a single Trimesh.Mesh object instead of a list. This can be useful for efficiency reasons if you know
                ahead of time all of the managed actors have the same collision mesh
        """
        assert (
            not self.merged
        ), "Currently you cannot fetch collision meshes of merged articulations as merged articulations only share a root link"
        if self.scene.gpu_sim_enabled:
            assert (
                self.scene._gpu_sim_initialized
            ), "During GPU simulation link pose data is not accessible until after \
                initialization, and link poses are needed to get the correct collision mesh of an entire articulation"
        else:
            self._objs[0].pose = self._objs[0].pose
        # TODO (stao): Can we have a batched version of trimesh?
        meshes: List[trimesh.Trimesh] = []

        for i, art in enumerate(self._objs):
            art_meshes = []
            for link in art.links:
                link_mesh = merge_meshes(get_component_meshes(link))
                if link_mesh is not None:
                    if to_world_frame:
                        pose = self.links[link.index].pose[i]
                        link_mesh.apply_transform(pose.sp.to_transformation_matrix())
                    art_meshes.append(link_mesh)
            mesh = merge_meshes(art_meshes)
            if mesh is not None:
                meshes.append(mesh)
            if first_only:
                break
        if len(meshes) == 0:
            return []
        if first_only:
            return meshes[0]
        return meshes

    def get_first_visual_mesh(self, to_world_frame: bool = True) -> trimesh.Trimesh:
        """
        Returns the visual mesh of the first managed articulation object. Note results of this are not cached or optimized at the moment
        so this function can be slow if called too often
        """
        return self.get_visual_meshes(to_world_frame=to_world_frame, first_only=True)

    def get_visual_meshes(
        self, to_world_frame: bool = True, first_only: bool = False
    ) -> List[trimesh.Trimesh]:
        """
        Returns the visual mesh of each managed articulation object. Note results of this are not cached or optimized at the moment
        so this function can be slow if called too often
        """
        assert (
            not self.merged
        ), "Currently you cannot fetch visual meshes of merged articulations as merged articulations only share a root link"
        if self.scene.gpu_sim_enabled:
            assert (
                self.scene._gpu_sim_initialized
            ), "During GPU simulation link pose data is not accessible until after \
                initialization, and link poses are needed to get the correct visual mesh of an entire articulation"
        else:
            self._objs[0].pose = self._objs[0].pose
        meshes: List[trimesh.Trimesh] = []
        for i, art in enumerate(self._objs):
            art_meshes = []
            for link in art.links:
                render_shapes = []
                rb_comp = link.entity.find_component_by_type(
                    sapien.render.RenderBodyComponent
                )
                if rb_comp is not None:
                    for render_shape in rb_comp.render_shapes:
                        render_shapes += get_render_shape_meshes(render_shape)
                    link_mesh = merge_meshes(render_shapes)
                    if link_mesh is not None:
                        if to_world_frame:
                            pose = self.links[link.index].pose[i]
                            link_mesh.apply_transform(
                                pose.sp.to_transformation_matrix()
                            )
                        art_meshes.append(link_mesh)
            mesh = merge_meshes(art_meshes)
            meshes.append(mesh)
            if first_only:
                break
        if first_only:
            return meshes[0]
        return meshes

    def get_net_contact_impulses(self, link_names: Union[List[str], Tuple[str]]):
        """Get net contact impulses for several links together. This should be faster compared to using
        link.get_net_contact_impulses on each link.

        Returns impulse vector of shape (N, len(link_names), 3) where N is the number of environments
        """
        if self.scene.gpu_sim_enabled:
            if tuple(link_names) not in self._net_contact_force_queries:
                bodies = []
                for k in link_names:
                    bodies += self.links_map[k]._bodies
                self._net_contact_force_queries[
                    tuple(link_names)
                ] = self.px.gpu_create_contact_body_impulse_query(bodies)
            query = self._net_contact_force_queries[tuple(link_names)]
            self.px.gpu_query_contact_body_impulses(query)
            return (
                query.cuda_impulses.torch()
                .clone()
                .reshape(len(link_names), -1, 3)
                .transpose(1, 0)
            )
        else:
            included_links = [self.links_map[k]._objs[0].entity for k in link_names]
            contacts = self.px.get_contacts()
            articulation_contacts = defaultdict(list)
            for contact in contacts:
                if contact.bodies[0].entity in included_links:
                    articulation_contacts[contact.bodies[0].entity.name].append(
                        (contact, True)
                    )
                elif contact.bodies[1].entity in included_links:
                    articulation_contacts[contact.bodies[1].entity.name].append(
                        (contact, False)
                    )

            net_impulse = torch.zeros(len(link_names), 3)
            for i, link_name in enumerate(link_names):
                link_contacts = articulation_contacts[link_name]
                if len(link_contacts) > 0:
                    total_impulse = np.zeros(3)
                    for contact, flag in link_contacts:
                        contact_impulse = np.sum(
                            [point.impulse for point in contact.points], axis=0
                        )
                        total_impulse += contact_impulse * (1 if flag else -1)
                    net_impulse[i] = common.to_tensor(total_impulse)
            return net_impulse[None, :]

    def get_net_contact_forces(self, link_names: Union[List[str], Tuple[str]]):
        """Get net contact forces for several links together. This should be faster compared to using
        link.get_net_contact_forces on each link.


        Returns force vector of shape (N, len(link_names), 3) where N is the number of environments
        """
        return self.get_net_contact_impulses(link_names) / self.scene.timestep

    def get_joint_target_indices(
        self, joint_indices: Union[Array, List[int], List[ArticulationJoint]]
    ):
        """
        Gets the meshgrid indexes for indexing px.cuda_articulation_target_* values given a 1D list of joint indexes or a 1D list of ArticulationJoint objects.

        Internally the given input is made to a tuple for a cache key and is used to cache results for fast lookup in the future, particularly for large-scale GPU simulations.
        """
        if isinstance(joint_indices, list):
            joint_indices = tuple(joint_indices)
        if joint_indices not in self._cached_joint_target_indices:
            vals = joint_indices
            if isinstance(joint_indices[0], ArticulationJoint):
                for joint in joint_indices:
                    assert (
                        joint.articulation == self
                    ), "Can only fetch this articulation's joint_target_indices when provided joints from this articulation"
                vals = [
                    x.active_index[0] for x in joint_indices
                ]  # active_index on joint is batched but it should be the same value across all managed joint objects
            self._cached_joint_target_indices[joint_indices] = torch.meshgrid(
                self._data_index,
                common.to_tensor(vals, device=self.device),
                indexing="ij",
            )
        return self._cached_joint_target_indices[joint_indices]

    def get_drive_targets(self):
        return self.drive_targets

    def get_drive_velocities(self):
        return self.drive_velocities

    @property
    def drive_targets(self):
        """
        The current drive targets of the active joints. Also known as the target joint positions. Returns a tensor
        of shape (N, M) where N is the number of environments and M is the number of active joints.
        """
        if self.scene.gpu_sim_enabled:
            return self.px.cuda_articulation_target_qpos.torch()[
                self.get_joint_target_indices(self.active_joints)
            ]
        else:
            return torch.cat([x.drive_target for x in self.active_joints], dim=-1)

    @property
    def drive_velocities(self):
        """
        The current drive velocity targets of the active joints. Also known as the target joint velocities. Returns a tensor
        of shape (N, M) where N is the number of environments and M is the number of active joints.
        """
        if self.scene.gpu_sim_enabled:
            return self.px.cuda_articulation_target_qvel.torch()[
                self.get_joint_target_indices(self.active_joints)
            ]
        else:
            return torch.cat(
                [x.drive_velocity_target for x in self.active_joints], dim=-1
            )

    # -------------------------------------------------------------------------- #
    # Functions from physx.PhysxArticulation
    # -------------------------------------------------------------------------- #
    def compute_passive_force(self, *args, **kwargs):
        if self.scene.gpu_sim_enabled:
            raise NotImplementedError(
                "Passive force computation is currently not supported in GPU PhysX"
            )
        else:
            return self._objs[0].compute_passive_force(*args, **kwargs)

    # def create_fixed_tendon(self, link_chain: list[PhysxArticulationLinkComponent], coefficients: list[float], recip_coefficients: list[float], rest_length: float = 0, offset: float = 0, stiffness: float = 0, damping: float = 0, low: float = -3.4028234663852886e+38, high: float = 3.4028234663852886e+38, limit_stiffness: float = 0) -> None: ...
    def find_joint_by_name(self, arg0: str) -> ArticulationJoint:
        if self.merged:
            raise RuntimeError(
                "Cannot call find_joint_by_name when the articulation object is managing articulations of different dofs"
            )
        return self.joints_map[arg0]

    def find_link_by_name(self, arg0: str) -> Link:
        if self.merged:
            raise RuntimeError(
                "Cannot call find_link_by_name when the articulation object is managing articulations of different dofs"
            )
        return self.links_map[arg0]

    def get_active_joints(self):
        return self.active_joints

    def get_dof(self) -> int:
        return self.dof

    # def get_gpu_index(self) -> int: ...
    def get_joints(self):
        return self.joints

    def get_link_incoming_joint_forces(self):
        """
        Returns the incoming joint forces, referred to as spatial forces, for each link, with shape (N, M, 6), where N is the number of environments and M is the number of links.

        Spatial forces are complex to describe and we recommend you see the physx documentation for how they are computed: https://nvidia-omniverse.github.io/PhysX/physx/5.6.0/docs/Articulations.html#link-incoming-joint-force

        The mth spatial force refers to the mth link, and to find a particular link you can find the Link object first then use it's index attribute.

        link = articulation.links_map[link_name]
        link_index = link.index
        link_incoming_joint_force = articulation.get_link_incoming_joint_forces()[:, link_index]

        """
        if self.scene.gpu_sim_enabled:
            self.px.gpu_fetch_articulation_link_incoming_joint_forces()
            # TODO (stao): we should lazy call the GPU data fetching functions in the future if necessarily. Since most users don't use this at the moment
            # we just fetch it live here instead of with all the other fetch functions
            return self.px.cuda_articulation_link_incoming_joint_forces.torch()[
                self._data_index, :
            ]
        else:
            return torch.from_numpy(
                self._objs[0].get_link_incoming_joint_forces()[None, :]
            )

    def get_links(self):
        return self.links

    def get_name(self) -> str:
        return self.name

    def get_pose(self) -> sapien.Pose:
        return self.pose

    # def get_qacc(self) -> numpy.ndarray[numpy.float32, _Shape[m, 1]]: ...
    def get_qf(self):
        return self.qf

    # def get_qlimit(self):
    # removed this function from ManiSkill Articulation wrapper API as it is redundant
    #     """
    #     same as get_qlimits
    #     """
    #     return self.qlimits

    def get_qlimits(self):
        return self.qlimits

    def get_qpos(self):
        return self.qpos

    def get_qvel(self):
        return self.qvel

    def get_root(self):
        return self.root

    def get_root_angular_velocity(self) -> torch.Tensor:
        return self.root_angular_velocity

    def get_root_linear_velocity(self) -> torch.Tensor:
        return self.root_linear_velocity

    def get_root_pose(self):
        return self.root_pose

    # def set_name(self, arg0: str) -> None: ...
    def set_pose(self, arg0: sapien.Pose) -> None:
        self.pose = arg0

    # def set_qacc(self, qacc: numpy.ndarray[numpy.float32, _Shape[m, 1]]) -> None: ...
    def set_qf(self, qf: Array) -> None:
        self.qf = qf

    def set_qpos(self, arg1: Array):
        self.qpos = arg1

    def set_qvel(self, qvel: Array) -> None:
        self.qvel = qvel

    def set_root_angular_velocity(self, velocity: Array) -> None:
        self.root_angular_velocity = velocity

    def set_root_linear_velocity(self, velocity: Array) -> None:
        self.root_linear_velocity = velocity

    def set_root_pose(self, pose: sapien.Pose) -> None:
        self.root_pose = pose

    # @property
    # def active_joints(self):
    #     return self._articulations[0].active_joints

    @cached_property
    def dof(self) -> torch.tensor:
        return torch.tensor([obj.dof for obj in self._objs], device=self.device)

    # @property
    # def gpu_index(self) -> int:
    #     """
    #     :type: int
    #     """
    # @property
    # def joints(self):
    #     return self._articulations[0].joints

    # @property
    # def links(self) -> list[PhysxArticulationLinkComponent]:
    #     """
    #     :type: list[PhysxArticulationLinkComponent]
    #     """
    # @property
    # def name(self) -> str:
    #     """
    #     :type: str
    #     """
    # @name.setter
    # def name(self, arg1: str) -> None:
    #     pass
    @property
    def pose(self) -> Pose:
        return self.root_pose

    @pose.setter
    def pose(self, arg1: Union[Pose, sapien.Pose]) -> None:
        self.root_pose = arg1

    @property
    def qacc(self):
        if self.scene.gpu_sim_enabled:
            return self.px.cuda_articulation_qacc.torch()[
                self._data_index, : self.max_dof
            ]
        else:
            return torch.from_numpy(self._objs[0].qacc[None, :])

    # @qacc.setter
    # def qacc(self, arg1: numpy.ndarray[numpy.float32, _Shape[m, 1]]) -> None:
    #     pass
    @property
    def qf(self):
        if self.scene.gpu_sim_enabled:
            return self.px.cuda_articulation_qf.torch()[
                self._data_index, : self.max_dof
            ]
        else:
            return torch.from_numpy(self._objs[0].qf[None, :])

    @qf.setter
    def qf(self, arg1: torch.Tensor):
        if self.scene.gpu_sim_enabled:
            arg1 = common.to_tensor(arg1, device=self.device)
            self.px.cuda_articulation_qf.torch()[
                self._data_index[self.scene._reset_mask[self._scene_idxs]],
                : self.max_dof,
            ] = arg1
        else:
            arg1 = common.to_numpy(arg1)
            if len(arg1.shape) == 2:
                arg1 = arg1[0]
            self._objs[0].qf = arg1

    @cached_property
    def qlimits(self):
        padded_qlimits = np.array(
            [
                np.concatenate([obj.qlimits, np.zeros((self.max_dof - obj.dof, 2))])
                for obj in self._objs
            ]
        )
        padded_qlimits = torch.from_numpy(padded_qlimits).float()
        return padded_qlimits.to(self.device)

    @property
    def qpos(self):
        if self.scene.gpu_sim_enabled:
            # NOTE (stao): cuda_articulation_qpos is of shape (M, N) where M is the total number of articulations in the physx scene,
            # N is the max dof of all those articulations.
            return self.px.cuda_articulation_qpos.torch()[
                self._data_index, : self.max_dof
            ]
        else:
            return torch.from_numpy(self._objs[0].qpos[None, :])

    @qpos.setter
    def qpos(self, arg1: torch.Tensor):
        if self.scene.gpu_sim_enabled:
            arg1 = common.to_tensor(arg1, device=self.device)
            self.px.cuda_articulation_qpos.torch()[
                self._data_index[self.scene._reset_mask[self._scene_idxs]],
                : self.max_dof,
            ] = arg1
        else:
            arg1 = common.to_numpy(arg1)
            if len(arg1.shape) == 2:
                arg1 = arg1[0]
            self._objs[0].qpos = arg1

    @property
    def qvel(self):
        if self.scene.gpu_sim_enabled:
            return self.px.cuda_articulation_qvel.torch()[
                self._data_index, : self.max_dof
            ]
        else:
            return torch.from_numpy(self._objs[0].qvel[None, :])

    @qvel.setter
    def qvel(self, arg1: torch.Tensor):
        if self.scene.gpu_sim_enabled:
            arg1 = common.to_tensor(arg1, device=self.device)
            self.px.cuda_articulation_qvel.torch()[
                self._data_index[self.scene._reset_mask[self._scene_idxs]],
                : self.max_dof,
            ] = arg1
        else:
            arg1 = common.to_numpy(arg1)
            if len(arg1.shape) == 2:
                arg1 = arg1[0]
            self._objs[0].qvel = arg1

    @property
    def root_angular_velocity(self) -> torch.Tensor:
        return self.root.angular_velocity

    @root_angular_velocity.setter
    def root_angular_velocity(self, arg1: Array) -> None:
        if self.scene.gpu_sim_enabled:
            arg1 = common.to_tensor(arg1, device=self.device)
            self.px.cuda_rigid_body_data.torch()[
                self.root._body_data_index[self.scene._reset_mask[self._scene_idxs]],
                7:10,
            ] = arg1
        else:
            arg1 = common.to_numpy(arg1)
            if len(arg1.shape) == 2:
                arg1 = arg1[0]
            self._objs[0].set_root_angular_velocity(arg1)

    @property
    def root_linear_velocity(self) -> torch.Tensor:
        return self.root.linear_velocity

    @root_linear_velocity.setter
    def root_linear_velocity(self, arg1: Array) -> None:
        if self.scene.gpu_sim_enabled:
            arg1 = common.to_tensor(arg1, device=self.device)
            self.px.cuda_rigid_body_data.torch()[
                self.root._body_data_index[self.scene._reset_mask[self._scene_idxs]],
                10:13,
            ] = arg1
        else:
            arg1 = common.to_numpy(arg1)
            if len(arg1.shape) == 2:
                arg1 = arg1[0]
            self._objs[0].set_root_linear_velocity(arg1)

    @property
    def root_pose(self):
        return self.root.pose

    @root_pose.setter
    def root_pose(self, arg1: Union[Pose, sapien.Pose]):
        self.root.pose = arg1

    # -------------------------------------------------------------------------- #
    # Functions not part of SAPIEN API provided for conveniency
    # -------------------------------------------------------------------------- #
    def create_pinocchio_model(self):
        # NOTE (stao): This is available but not typed in SAPIEN
        if self.scene.gpu_sim_enabled:
            raise NotImplementedError(
                "Cannot create a pinocchio model when GPU is enabled."
            )
        else:
            return self._objs[0].create_pinocchio_model()

    def set_joint_drive_targets(
        self,
        targets: Array,
        joints: Optional[List[ArticulationJoint]] = None,
        joint_indices: Optional[torch.Tensor] = None,
    ):
        """
        Set drive targets on active joints given joints. For GPU simulation only joint_indices argument is supported, which should be a 1D list of the active joint indices in the articulation.

        joints argument will always be used when possible and is the recommended approach. On CPU simulation the joints argument is required, joint_indices is not supported.
        """
        if self.scene.gpu_sim_enabled:
            targets = common.to_tensor(targets, device=self.device)
            if joints is not None:
                gx, gy = self.get_joint_target_indices(joints)
            else:
                gx, gy = self.get_joint_target_indices(joint_indices)
            self.px.cuda_articulation_target_qpos.torch()[
                gx[self.scene._reset_mask[self._scene_idxs]], gy[self.scene._reset_mask[self._scene_idxs]]
            ] = targets
        else:
            for i, joint in enumerate(joints):
                joint.set_drive_target(targets[0, i])

    def set_joint_drive_velocity_targets(
        self,
        targets: Array,
        joints: Optional[List[ArticulationJoint]] = None,
        joint_indices: Optional[torch.Tensor] = None,
    ):
        """
        Set drive velocity targets on active joints given joints. For GPU simulation only joint_indices argument is supported, which should be a 1D list of the active joint indices in the articulation.

        joints argument will always be used when possible and is the recommended approach. On CPU simulation the joints argument is required, joint_indices is not supported.
        """
        if self.scene.gpu_sim_enabled:
            targets = common.to_tensor(targets, device=self.device)
            if joints is not None:
                gx, gy = self.get_joint_target_indices(joints)
            else:
                gx, gy = self.get_joint_target_indices(joint_indices)
            self.px.cuda_articulation_target_qvel.torch()[
                gx[self.scene._reset_mask[self._scene_idxs]], gy[self.scene._reset_mask[self._scene_idxs]]
            ] = targets
        else:
            for i, joint in enumerate(joints):
                joint.set_drive_velocity_target(targets[0, i])
