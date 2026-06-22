from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Generic, TypeVar, cast

import numpy as np
import sapien.physx as physx
import torch

from mani_skill.sim.sapien.structs.decorators import before_gpu_init
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs import Array, Pose
from mani_skill.utils.structs.base import BaseStruct

if TYPE_CHECKING:
    from mani_skill.sim.sapien.sim import SapienSim

T = TypeVar("T")


@dataclass
class SapienBaseStruct(BaseStruct, Generic[T]):
    """
    Base class for all structs that manage objects in simulation across sub-scenes.
    """

    sim: SapienSim
    """The Sapien simulation backend object the struct operates on."""

    _objs: list[T]
    """
    list of objects of type T managed by this dataclass. This should not be modified after
    initialization. The struct hash is dependent on the hash of this list.
    """

    def __str__(self):
        return (
            f"<struct of type {self.__class__}; managing {self._num_objs} "
            f"{self._objs[0].__class__} objects>"
        )

    @cached_property
    def __maniskill_hash__(self):
        """A better hash to use compared to the default frozen dataclass hash.
        It is tied directly to the only immutable field (the _objs list)."""
        return hash(tuple([obj.__hash__() for obj in self._objs]))

    @property
    def px(self):
        """The physx system objects managed by this dataclass are working on"""
        return self.sim.px


@dataclass
class PhysxRigidBaseComponentStruct(SapienBaseStruct[T], Generic[T]):
    _bodies: list[physx.PhysxRigidBaseComponent]

    # ---------------------------------------------------------------------------- #
    # API from physx.PhysxRigidBaseComponent
    # ---------------------------------------------------------------------------- #
    # TODO (stao): To be added
    # def attach(self, collision_shape: PhysxCollisionShape)
    #     ...
    # def compute_global_aabb_tight(self)
    #     ...
    # def get_collision_shapes(self)
    #     ...
    # def get_global_aabb_fast(self)
    #     ...
    # @property
    # def _physx_pointer(self) -> int:
    #     ...
    # @property
    # def collision_shapes(self)
    #     ...


@dataclass
class PhysxRigidBodyComponentStruct(PhysxRigidBaseComponentStruct[T], Generic[T]):
    _bodies: list[physx.PhysxRigidBodyComponent]
    _body_data_name: str | None = None
    _body_data_index_internal: torch.Tensor | None = None

    @property
    def px(self):
        """The physx system objects managed by this dataclass are working on"""
        return self.sim.px

    @cached_property
    def _body_data_index(self):
        """a list of indexes of each GPU rigid body in the `px.cuda_rigid_body_data` buffer, one for
        each element in `self._objs`"""
        if self._body_data_index_internal is None:
            self._body_data_index_internal = torch.tensor(
                [
                    cast(physx.PhysxRigidDynamicComponent, body).gpu_pose_index
                    for body in self._bodies
                ],
                device=self.device,
            )
        return self._body_data_index_internal

    @property
    def _body_data(self) -> torch.Tensor:
        return getattr(self.px, self._body_data_name).torch()  # type: ignore

    @cached_property
    def _body_force_query(self):
        return cast(
            physx.PhysxGpuSystem, self.px
        ).gpu_create_contact_body_impulse_query(self._bodies)  # type: ignore

    def get_net_contact_forces(self):
        """
        Get the net contact forces on this body. Returns force vector of shape (N, 3)
        where N is the number of environments, and 3 is the dimension of the force vector itself,
        representing x, y, and z direction of force.
        """
        return self.get_net_contact_impulses() / self.sim.timestep

    def get_net_contact_impulses(self):
        """
        Get the net contact impulses on this body. Returns impulse vector of shape (N, 3)
        where N is the number of environments, and 3 is the dimension of the impulse vector itself,
        representing x, y, and z direction of impulse.
        """
        if self.sim.gpu_sim_enabled:
            cast(physx.PhysxGpuSystem, self.px).gpu_query_contact_body_impulses(
                self._body_force_query
            )
            return self._body_force_query.cuda_impulses.torch().clone()
        else:
            body_contacts = sapien_utils.get_cpu_actor_contacts(
                cast(physx.PhysxCpuSystem, self.px).get_contacts(),
                self._bodies[0].entity,
            )
            net_force = common.to_tensor(
                sapien_utils.compute_total_impulse(body_contacts)
            )
            return net_force[None, :]

    # ---------------------------------------------------------------------------- #
    # API from physx.PhysxRigidBodyComponent
    # ---------------------------------------------------------------------------- #

    # TODO: To be added
    # def add_force_at_point
    # def add_force_torque
    def get_angular_damping(self) -> torch.Tensor:
        return self.angular_damping

    def get_angular_velocity(self) -> torch.Tensor:
        return self.angular_velocity

    def get_auto_compute_mass(self) -> torch.Tensor:
        return self.auto_compute_mass

    # def get_cmass_local_pose(self):
    #     return

    def get_disable_gravity(self) -> torch.Tensor:
        return self.disable_gravity

    # def get_inertia(self) -> numpy.ndarray[numpy.float32, _Shape, _Shape[3]]: ...
    def get_linear_damping(self) -> torch.Tensor:
        return self.linear_damping

    def get_linear_velocity(self) -> torch.Tensor:
        return self.linear_velocity

    def get_mass(self) -> torch.Tensor:
        return self.mass

    # def get_max_contact_impulse(self) -> float: ... # TODO (Stao)
    # def get_max_depenetraion_velocity(self) -> float: ... # TODO (Stao)
    def set_angular_damping(self, damping: float) -> None:
        self.angular_damping = damping

    # def set_cmass_local_pose(self, arg0: sapien.pysapien.Pose) -> None: ...
    def set_disable_gravity(self, arg0: bool) -> None:
        self.disable_gravity = arg0

    # def set_inertia(self, arg0: numpy.ndarray[numpy.float32, _Shape, _Shape[3]]) -> None: ...
    def set_linear_damping(self, damping: float) -> None:
        self.linear_damping = damping

    def set_mass(self, arg0: float) -> None:
        self.mass = arg0

    # def set_max_contact_impulse(self, impulse: float) -> None: ... # TODO (Stao)
    # def set_max_depenetraion_velocity(self, velocity: float) -> None: ... # TODO (Stao)
    @property
    def angular_damping(self) -> torch.Tensor:
        return torch.tensor([body.angular_damping for body in self._bodies])

    @angular_damping.setter
    @before_gpu_init
    def angular_damping(self, arg1: float) -> None:
        for rb in self._bodies:
            rb.angular_damping = arg1

    @property
    def angular_velocity(self) -> torch.Tensor:
        if self.sim.gpu_sim_enabled:
            return self._body_data[self._body_data_index, 10:13]
        else:
            return torch.tensor(
                np.array([body.angular_velocity for body in self._bodies]),
                device=self.device,
            )

    @property
    def auto_compute_mass(self) -> torch.Tensor:
        return torch.tensor([body.auto_compute_mass for body in self._bodies])

    @cached_property
    def cmass_local_pose(self) -> Pose:
        raw_poses = np.stack(
            [
                np.concatenate([x.cmass_local_pose.p, x.cmass_local_pose.q])
                for x in self._bodies
            ]
        )
        return Pose.create(common.to_tensor(raw_poses), device=self.device)

    # @cmass_local_pose.setter
    # def cmass_local_pose(self, arg1: sapien.pysapien.Pose) -> None:
    #     pass
    @property
    def disable_gravity(self) -> torch.Tensor:
        return torch.tensor([body.disable_gravity for body in self._bodies])

    @disable_gravity.setter
    @before_gpu_init
    def disable_gravity(self, arg1: bool) -> None:
        for rb in self._bodies:
            rb.disable_gravity = arg1

    # @property
    # def inertia(self) -> numpy.ndarray[numpy.float32, _Shape, _Shape[3]]:
    #     """
    #     :type: numpy.ndarray[numpy.float32, _Shape, _Shape[3]]
    #     """
    # @inertia.setter
    # def inertia(self, arg1: numpy.ndarray[numpy.float32, _Shape, _Shape[3]]) -> None:
    #     pass
    @property
    def linear_damping(self) -> torch.Tensor:
        return torch.tensor([body.linear_damping for body in self._bodies])

    @linear_damping.setter
    @before_gpu_init
    def linear_damping(self, arg1: float) -> None:
        for rb in self._bodies:
            rb.linear_damping = arg1

    @property
    def linear_velocity(self) -> torch.Tensor:
        if self.sim.gpu_sim_enabled:
            return self._body_data[self._body_data_index, 7:10]
        else:
            return torch.from_numpy(self._bodies[0].linear_velocity[None, :]).to(
                self.device
            )

    @property
    def mass(self) -> torch.Tensor:
        return torch.tensor([body.mass for body in self._bodies])

    @mass.setter
    @before_gpu_init
    def mass(self, arg1: float) -> None:
        for body in self._bodies:
            body.set_mass(arg1)

    # @property
    # def max_contact_impulse(self) -> float:
    #     """
    #     :type: float
    #     """
    # @max_contact_impulse.setter
    # def max_contact_impulse(self, arg1: float) -> None:
    #     pass
    # @property
    # def max_depenetraion_velocity(self) -> float:
    #     """
    #     :type: float
    #     """
    # @max_depenetraion_velocity.setter
    # def max_depenetraion_velocity(self, arg1: float) -> None:
    #     pass
    # pass


@dataclass
class PhysxRigidDynamicComponentStruct(PhysxRigidBodyComponentStruct[T], Generic[T]):
    _bodies: list[physx.PhysxRigidDynamicComponent]

    def get_angular_velocity(self) -> torch.Tensor:
        return self.angular_velocity

    def get_gpu_index(self) -> list[int]:
        return self.gpu_index

    def get_gpu_pose_index(self) -> list[int]:
        return self.gpu_pose_index

    # def get_kinematic(self) -> bool:
    #     return self.kinematic
    # def get_kinematic_target(self) -> sapien.pysapien.Pose: ...
    def get_linear_velocity(self) -> torch.Tensor:
        return self.linear_velocity

    # NOTE (fxiang): Cannot lock after gpu setup
    def get_locked_motion_axes(self) -> Array:
        return self.locked_motion_axes

    # def put_to_sleep(self) -> None: ...
    def set_angular_velocity(self, arg0: Array):
        """
        Set the angular velocity of the dynamic rigid body.

        Args:
            arg0: The angular velocity to set. Can be of shape (N, 3) where N is the number of
                managed bodies or (3,) to apply the same angular velocity to all managed bodies.
        """
        self.angular_velocity = arg0

    # def set_kinematic(self, arg0: bool) -> None: ...
    # def set_kinematic_target(self, arg0: sapien.pysapien.Pose) -> None: ...
    def set_linear_velocity(self, arg0: Array):
        """
        Set the linear velocity of the dynamic rigid body.
        Args:
            arg0: The linear velocity to set. Can be of shape (N, 3) where N is the number
                of managed bodies or (3,) to apply the same linear velocity to all managed
                bodies.
        """
        self.linear_velocity = arg0

    def set_locked_motion_axes(self, axes: Array) -> None:
        """
        Set some motion axes of the dynamic rigid body to be locked
        Args:
            axes: list of 6 true/false values indicating which of the 6 DOFs of the body is
                locked. The order is linear X, Y, Z followed by angular X, Y, Z. If given a
                single list of length 6, it will be applied to all managed bodies. If given a
                batch of shape (N, 6), you can modify the N managed bodies each in batch.


        Example:
            set_locked_motion_axes([True, False, False, False, True, False]) allows the object
            to move along the X axis and rotate about the Y axis.
        """
        self.locked_motion_axes = axes

    # def wake_up(self) -> None: ...
    @property
    def angular_velocity(self) -> torch.Tensor:
        if self.sim.gpu_sim_enabled:
            return self._body_data[self._body_data_index, 10:13]
        else:
            return torch.from_numpy(self._bodies[0].angular_velocity[None, :]).to(
                self.device
            )

    @angular_velocity.setter
    def angular_velocity(self, arg1: Array):
        if self.sim.gpu_sim_enabled:
            arg1 = common.to_tensor(arg1, device=self.device)
            self._body_data[
                self._body_data_index[self.sim.scene._reset_mask[self._scene_idxs]],
                10:13,
            ] = arg1
        else:
            arg1 = common.to_numpy(arg1)
            if len(arg1.shape) == 2:
                arg1 = arg1[0]
            self._bodies[0].angular_velocity = arg1  # type: ignore

    @property
    def gpu_index(self):
        if self.sim.gpu_sim_enabled:
            return [b.gpu_index for b in self._bodies]
        else:
            raise AttributeError("GPU index is not supported when gpu is not enabled")

    @property
    def gpu_pose_index(self):
        if self.sim.gpu_sim_enabled:
            return [b.gpu_pose_index for b in self._bodies]
        else:
            raise AttributeError(
                "GPU pose index is not supported when gpu is not enabled"
            )

    @property
    @before_gpu_init
    def is_sleeping(self):
        if self.sim.gpu_sim_enabled:
            return [b.is_sleeping for b in self._bodies]
        else:
            return [self._bodies[0].is_sleeping]

    # @property
    # def kinematic(self) -> bool:
    #     """
    #     :type: bool
    #     """
    #     if self.px_body_type == "static": return False

    #     return self._bodies[0].kinematic  # note that all bodies must of the same type

    # @kinematic.setter
    # def kinematic(self, arg1: bool) -> None:
    #     if physx.is_gpu_enabled():
    #         raise NotImplementedError("Cannot change kinematic of body in GPU mode")
    #     else:
    #         self._bodies[0].kinematic = arg1

    # @property
    # def kinematic_target(self) -> sapien.pysapien.Pose:
    #     """
    #     :type: sapien.pysapien.Pose
    #     """
    # @kinematic_target.setter
    # def kinematic_target(self, arg1: sapien.pysapien.Pose) -> None:
    #     pass
    @property
    def linear_velocity(self) -> torch.Tensor:
        if self.sim.gpu_sim_enabled:
            return self._body_data[self._body_data_index, 7:10]
        else:
            return torch.tensor(
                np.array([body.linear_velocity for body in self._bodies]),
                device=self.device,
            )

    @linear_velocity.setter
    def linear_velocity(self, arg1: Array):
        if self.sim.gpu_sim_enabled:
            arg1 = common.to_tensor(arg1, device=self.device)
            self._body_data[
                self._body_data_index[self.sim.scene._reset_mask[self._scene_idxs]],
                7:10,
            ] = arg1
        else:
            arg1 = common.to_numpy(arg1)
            if len(arg1.shape) == 2:
                arg1 = arg1[0]
            self._bodies[0].linear_velocity = arg1  # type: ignore

    @property
    def locked_motion_axes(self) -> Array:
        """
        :type: list[bool]
        """
        return torch.tensor(
            [body.locked_motion_axes for body in self._bodies], device=self.device
        )

    @locked_motion_axes.setter
    @before_gpu_init
    def locked_motion_axes(self, arg1: Array) -> None:
        arg1 = common.to_tensor(arg1, device=self.device)
        if arg1.shape[0] == 6:
            for body in self._bodies:
                body.set_locked_motion_axes(arg1.cpu().tolist())
        else:
            for i, body in enumerate(self._bodies):
                body.set_locked_motion_axes(arg1[i].cpu().tolist())


@dataclass
class PhysxJointComponentStruct(SapienBaseStruct[T], Generic[T]):
    # parent: PhysxRigidBaseComponentStruct # TODO what is this for?
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
