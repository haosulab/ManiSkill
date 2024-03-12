from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, List, Tuple, Union

import numpy as np
import sapien
import sapien.physx as physx
import torch
import trimesh

from mani_skill.utils import sapien_utils
from mani_skill.utils.geometry.trimesh_utils import get_component_meshes, merge_meshes
from mani_skill.utils.structs.base import BaseStruct
from mani_skill.utils.structs.joint import Joint
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.pose import Pose
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
    link_map: OrderedDict[str, Link]
    """Maps link name to the Link object"""
    root: Link
    """The root Link object"""
    joints: List[Joint]
    """List of Joint objects"""
    joint_map: OrderedDict[str, Joint]
    """Maps joint name to the Joint object"""
    active_joints: List[Joint]
    """List of active Joint objects, referencing elements in self.joints"""
    active_joint_map: OrderedDict[str, Joint]
    """Maps active joint name to the Joint object, referencing elements in self.joints"""

    name: str = None
    """Name of this articulation"""

    _merged: bool = False
    """
    whether or not this articulation object is a merged articulation where it is managing many articulations with different dofs.

    There are a number of caveats when it comes to merged articulations. While merging articulations means you can easily fetch padded
    qpos, qvel, etc. type data, a number of attributes and functions will make little sense and you should avoid using them unless you
    are an advanced user. In particular, the list of Links, Joints, their corresponding maps, net contact forces of multiple links, no
    longer make "sense"
    """

    _cached_joint_target_indices: OrderedDict[int, torch.Tensor] = field(
        default_factory=OrderedDict
    )
    """Map from a set of joints of this articulation and the indexing torch tensor to use for setting drive targets"""

    _net_contact_force_queries: OrderedDict[
        Tuple, physx.PhysxGpuContactBodyImpulseQuery
    ] = field(default_factory=OrderedDict)
    """Maps a tuple of link names to pre-saved net contact force queries"""

    @classmethod
    def create_from_physx_articulations(
        cls,
        physx_articulations: List[physx.PhysxArticulation],
        scene: ManiSkillScene,
        scene_idxs: torch.Tensor,
        _merged: bool = False,
    ):
        shared_name = "_".join(physx_articulations[0].name.split("_")[1:])

        # NOTE (stao): This is a little bit of a non-standard way to use @classmethod style creation functions for dataclasses
        # however generating the Joint and Link objects relies on a reference to the articulation properties so this is done as a
        # convenience
        self = cls(
            _objs=physx_articulations,
            _scene=scene,
            _scene_idxs=scene_idxs,
            links=[],
            link_map=OrderedDict(),
            root=None,
            joints=[],
            joint_map=OrderedDict(),
            active_joints=[],
            active_joint_map=OrderedDict(),
            name=shared_name,
            _merged=_merged,
        )
        # create link and joint structs
        # num_links = len(physx_articulations[0].links)
        num_links = max([len(x.links) for x in physx_articulations])
        all_links_objs: List[List[physx.PhysxArticulationLinkComponent]] = [
            [] for _ in range(num_links)
        ]
        # num_joints = len(physx_articulations[0].joints)
        num_joints = max([len(x.joints) for x in physx_articulations])
        all_joint_objs: List[List[physx.PhysxArticulationJoint]] = [
            [] for _ in range(num_joints)
        ]

        link_map = OrderedDict()
        for articulation in physx_articulations:
            # assert num_links == len(articulation.links) and num_joints == len(
            #     articulation.joints
            # ), "Gave different physx articulations. Each Articulation object can only manage the same articulations, not different ones"
            for i, link in enumerate(articulation.links):
                all_links_objs[i].append(link)
            for i, joint in enumerate(articulation.joints):
                all_joint_objs[i].append(joint)
        wrapped_links: List[Link] = []
        for links in all_links_objs:
            wrapped_link = Link.create(links, self)
            link_map[wrapped_link.name] = wrapped_link
            wrapped_links.append(wrapped_link)
            assert wrapped_link.is_root.any() == wrapped_link.is_root.all()
            if wrapped_link.is_root.any():
                root = wrapped_link
        self.links = wrapped_links
        self.link_map = link_map
        self.root = root

        # create Joint objects and figure out active joint references
        all_active_joint_names = [
            x.name for x in physx_articulations[0].get_active_joints()
        ]
        all_joint_names = [x.name for x in physx_articulations[0].joints]
        active_joint_indices = [
            all_joint_names.index(x) for x in all_active_joint_names
        ]

        joint_map = OrderedDict()
        wrapped_joints: List[Joint] = []
        for joint_index, joints in enumerate(all_joint_objs):
            try:
                active_joint_index = all_active_joint_names.index(
                    all_joint_names[joint_index]
                )
            except:
                active_joint_index = None
            wrapped_joint = Joint.create(joints, self, joint_index, active_joint_index)
            joint_map[wrapped_joint.name] = wrapped_joint
            wrapped_joints.append(wrapped_joint)
        self.joints = wrapped_joints
        self.joint_map = joint_map
        self.active_joints = [wrapped_joints[i] for i in active_joint_indices]
        self.active_joint_map = {joint.name: joint for joint in self.active_joints}
        return self

    @classmethod
    def merge(cls, articulations: List["Articulation"], name: str = None):
        objs = []
        scene = articulations[0]._scene
        merged_scene_idxs = []
        num_objs_per_actor = articulations[0]._num_objs
        for articulation in articulations:
            objs += articulation._objs
            merged_scene_idxs.append(articulation._scene_idxs)
            del scene.articulations[articulation.name]
            assert (
                articulation._num_objs == num_objs_per_actor
            ), "Each given articulation must have the same number of managed objects"
        merged_scene_idxs = torch.concat(merged_scene_idxs)
        merged_articulation = Articulation.create_from_physx_articulations(
            objs, scene, merged_scene_idxs, _merged=True
        )
        merged_articulation.name = name
        scene.articulations[merged_articulation.name] = merged_articulation
        return merged_articulation

    # -------------------------------------------------------------------------- #
    # Additional useful functions not in SAPIEN original API
    # -------------------------------------------------------------------------- #

    @cached_property
    def _data_index(self):
        return torch.tensor(
            [px_articulation.gpu_index for px_articulation in self._objs],
            device=self.device,
            dtype=torch.int32,
        )

    def get_state(self):
        pose = self.root.pose
        vel = self.root.get_linear_velocity()  # [N, 3]
        ang_vel = self.root.get_angular_velocity()  # [N, 3]
        qpos = self.get_qpos()
        qvel = self.get_qvel()
        return torch.hstack([pose.p, pose.q, vel, ang_vel, qpos, qvel])

    def set_state(self, state: Array):
        if physx.is_gpu_enabled():
            state = sapien_utils.to_tensor(state)
            self.set_root_pose(Pose.create(state[:, :7]))
            self.set_root_linear_velocity(state[:, 7:10])
            self.set_root_angular_velocity(state[:, 10:13])
            # TODO (stao): Handle get/set state for envs with different DOFs. Perhaps need to let user set a padding ahead of time to ensure state is the same?
            self.set_qpos(state[:, 13 : 13 + self.max_dof])
            self.set_qvel(state[:, 13 + self.max_dof :])
        else:
            state = sapien_utils.to_numpy(state[0])
            self.set_root_pose(sapien.Pose(state[0:3], state[3:7]))
            self.set_root_linear_velocity(state[7:10])
            self.set_root_angular_velocity(state[10:13])
            qpos, qvel = np.split(state[13:], 2)
            self.set_qpos(qpos)
            self.set_qvel(qvel)

    @cached_property
    def max_dof(self) -> int:
        return max([obj.dof for obj in self._objs])

    # currently does not work in SAPIEN. Must set collision groups prior to building Link
    # def disable_self_collisions(self):
    #     """Disable all self collisions between links. Note that 1 << 31 is reserved in the entirety of ManiSkill for link self collisions"""
    #     for link in self.links:
    #         for obj in link._objs:
    #             for s in obj.get_collision_shapes():
    #                 g0, g1, g2, g3 = s.get_collision_groups()
    #                 s.set_collision_groups([g0, g1, g2 | (1 << 29), g3])

    def get_collision_mesh(
        self, to_world_frame: bool = True, first_only: bool = True
    ) -> trimesh.Trimesh:
        """
        Returns the collision mesh of each managed articulation object. Results of this are cached

        TODO (stao): Can we have a batched version of trimesh?
        """
        meshes = []
        mat = self.pose.to_transformation_matrix()
        for i, art in enumerate(self._objs):
            art_meshes = []
            for link in art.links:
                link_mesh = merge_meshes(get_component_meshes(link))
                if link_mesh is not None:
                    if to_world_frame:
                        link_mesh.apply_transform(link.pose.to_transformation_matrix())
                    art_meshes.append(link_mesh)
            mesh = merge_meshes(art_meshes)
            if to_world_frame:
                mesh.apply_transform(mat[i])
            meshes.append(mesh)
            if first_only:
                break
        return meshes

    def get_net_contact_forces(self, link_names: Union[List[str], Tuple[str]]):
        """Get net contact forces for several links together. This should be faster compared to using
        link.get_net_contact_forces on each link.


        Returns torch.Tensor of shape (num_envs, len(link_names), 3)
        """

        if physx.is_gpu_enabled():
            if tuple(link_names) not in self._net_contact_force_queries:
                bodies = []
                for k in link_names:
                    bodies += self.link_map[k]._bodies
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
                / self._scene.timestep
            )
        else:

            body_contacts = sapien_utils.get_articulation_contacts(
                self.px.get_contacts(),
                self._objs[0],
                included_links=[self.link_map[k]._objs[0] for k in link_names],
            )
            net_force = (
                sapien_utils.to_tensor(
                    sapien_utils.compute_total_impulse(body_contacts)
                )
                / self._scene.timestep
            )
            return net_force[None, :]

    # -------------------------------------------------------------------------- #
    # Functions from physx.PhysxArticulation
    # -------------------------------------------------------------------------- #
    def compute_passive_force(self, *args, **kwargs):
        if physx.is_gpu_enabled():
            raise NotImplementedError(
                "Passive force computation is currently not supported in PhysX"
            )
        else:
            return self._objs[0].compute_passive_force(*args, **kwargs)

    # def create_fixed_tendon(self, link_chain: list[PhysxArticulationLinkComponent], coefficients: list[float], recip_coefficients: list[float], rest_length: float = 0, offset: float = 0, stiffness: float = 0, damping: float = 0, low: float = -3.4028234663852886e+38, high: float = 3.4028234663852886e+38, limit_stiffness: float = 0) -> None: ...
    def find_joint_by_name(self, arg0: str) -> Joint:
        if self._merged:
            raise RuntimeError(
                "Cannot call find_joint_by_name when the articulation object is managing articulations of different dofs"
            )
        return self.joint_map[arg0]

    def find_link_by_name(self, arg0: str) -> Link:
        if self._merged:
            raise RuntimeError(
                "Cannot call find_link_by_name when the articulation object is managing articulations of different dofs"
            )
        return self.link_map[arg0]

    def get_active_joints(self):
        return self.active_joints

    def get_dof(self) -> int:
        return self.dof

    # def get_gpu_index(self) -> int: ...
    def get_joints(self):
        return self.joints

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
        return torch.tensor([obj.dof for obj in self._objs])

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

    # @property
    # def qacc(self) -> numpy.ndarray[numpy.float32, _Shape[m, 1]]:
    #     """
    #     :type: numpy.ndarray[numpy.float32, _Shape[m, 1]]
    #     """
    # @qacc.setter
    # def qacc(self, arg1: numpy.ndarray[numpy.float32, _Shape[m, 1]]) -> None:
    #     pass
    @property
    def qf(self):
        if physx.is_gpu_enabled():
            return self.px.cuda_articulation_qf.torch()[
                self._data_index, : self.max_dof
            ]
        else:
            return torch.from_numpy(self._objs[0].qf[None, :])

    @qf.setter
    def qf(self, arg1: torch.Tensor):
        if physx.is_gpu_enabled():
            arg1 = sapien_utils.to_tensor(arg1)
            self.px.cuda_articulation_qf.torch()[
                self._data_index[self._scene._reset_mask], : self.max_dof
            ] = arg1
        else:
            arg1 = sapien_utils.to_numpy(arg1)
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
        if physx.is_gpu_enabled():
            return padded_qlimits.cuda()
        else:
            return padded_qlimits

    @property
    def qpos(self):
        if physx.is_gpu_enabled():
            # NOTE (stao): cuda_articulation_qpos is of shape (M, N) where M is the total number of articulations in the physx scene,
            # N is the max dof of all those articulations.
            return self.px.cuda_articulation_qpos.torch()[
                self._data_index, : self.max_dof
            ]
        else:
            return torch.from_numpy(self._objs[0].qpos[None, :])

    @qpos.setter
    def qpos(self, arg1: torch.Tensor):
        if physx.is_gpu_enabled():
            arg1 = sapien_utils.to_tensor(arg1)
            self.px.cuda_articulation_qpos.torch()[
                self._data_index[self._scene._reset_mask], : self.max_dof
            ] = arg1
        else:
            arg1 = sapien_utils.to_numpy(arg1)
            if len(arg1.shape) == 2:
                arg1 = arg1[0]
            self._objs[0].qpos = arg1

    @property
    def qvel(self):
        if physx.is_gpu_enabled():
            return self.px.cuda_articulation_qvel.torch()[
                self._data_index, : self.max_dof
            ]
        else:
            return torch.from_numpy(self._objs[0].qvel[None, :])

    @qvel.setter
    def qvel(self, arg1: torch.Tensor):
        if physx.is_gpu_enabled():
            arg1 = sapien_utils.to_tensor(arg1)
            self.px.cuda_articulation_qvel.torch()[
                self._data_index[self._scene._reset_mask], : self.max_dof
            ] = arg1
        else:
            arg1 = sapien_utils.to_numpy(arg1)
            if len(arg1.shape) == 2:
                arg1 = arg1[0]
            self._objs[0].qvel = arg1

    @property
    def root_angular_velocity(self) -> torch.Tensor:
        return self.root.angular_velocity

    @root_angular_velocity.setter
    def root_angular_velocity(self, arg1: Array) -> None:
        if physx.is_gpu_enabled():
            arg1 = sapien_utils.to_tensor(arg1)
            self.px.cuda_rigid_body_data.torch()[
                self.root._body_data_index[self._scene._reset_mask], 10:13
            ] = arg1
        else:
            arg1 = sapien_utils.to_numpy(arg1)
            if len(arg1.shape) == 2:
                arg1 = arg1[0]
            self._objs[0].set_root_angular_velocity(arg1)

    @property
    def root_linear_velocity(self) -> torch.Tensor:
        return self.root.linear_velocity

    @root_linear_velocity.setter
    def root_linear_velocity(self, arg1: Array) -> None:
        if physx.is_gpu_enabled():
            arg1 = sapien_utils.to_tensor(arg1)
            self.px.cuda_rigid_body_data.torch()[
                self.root._body_data_index[self._scene._reset_mask], 7:10
            ] = arg1
        else:
            arg1 = sapien_utils.to_numpy(arg1)
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
        if physx.is_gpu_enabled():
            raise NotImplementedError(
                "Cannot create a pinocchio model when GPU is enabled. If you wish to do inverse kinematics you must use pytorch_kinematics"
            )
        else:
            return self._objs[0].create_pinocchio_model()

    # def _get_joint_indices(self, joints: List[Joint]):
    #     if

    def set_joint_drive_targets(
        self,
        targets: Array,
        joints: List[Joint] = None,
        joint_indices: torch.Tensor = None,
    ):
        """
        Set drive targets on active joints. Joint indices are required to be given for GPU sim, and joint objects are required for the CPU sim

        TODO (stao): can we use joint indices for the CPU sim as well? Some global joint indices?
        """
        if physx.is_gpu_enabled():
            targets = sapien_utils.to_tensor(targets)
            if joint_indices not in self._cached_joint_target_indices:
                self._cached_joint_target_indices[joint_indices] = torch.meshgrid(
                    self._data_index, joint_indices, indexing="ij"
                )
            gx, gy = self._cached_joint_target_indices[joint_indices]
            self.px.cuda_articulation_target_qpos.torch()[gx, gy] = targets
        else:
            for i, joint in enumerate(joints):
                joint.set_drive_target(targets[0, i])

    def set_joint_drive_velocity_targets(
        self,
        targets: Array,
        joints: List[Joint] = None,
        joint_indices: torch.Tensor = None,
    ):
        """
        Set drive velocity targets on active joints. Joint indices are required to be given for GPU sim, and joint objects are required for the CPU sim

        TODO (stao): can we use joint indices for the CPU sim as well? Some global joint indices?
        """
        if physx.is_gpu_enabled():
            targets = sapien_utils.to_tensor(targets)
            if joint_indices not in self._cached_joint_target_indices:
                self._cached_joint_target_indices[joint_indices] = torch.meshgrid(
                    self._data_index, joint_indices, indexing="ij"
                )
            gx, gy = self._cached_joint_target_indices[joint_indices]
            self.px.cuda_articulation_target_qvel.torch()[gx, gy] = targets
        else:
            for i, joint in enumerate(joints):
                joint.set_drive_velocity_target(targets[0, i])
