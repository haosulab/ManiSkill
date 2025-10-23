from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
import sapien.physx as physx
import torch

from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.decorators import before_gpu_init
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
T = TypeVar("T")


@dataclass
class BaseStruct(Generic[T]):
    """
    Base class of all structs that manage sapien objects on CPU/GPU
    """

    _objs: list[T]
    """list of objects of type T managed by this dataclass. This should not be modified after initialization. The struct hash is dependent on the hash of this list."""
    _scene_idxs: torch.Tensor
    """a list of indexes parallel to `self._objs` indicating which sub-scene each of those objects are actually in by index"""
    scene: ManiSkillScene
    """The ManiSkillScene object that manages the sub-scenes this dataclasses's objects are in"""

    def __post_init__(self):
        if not isinstance(self._scene_idxs, torch.Tensor):
            self._scene_idxs = common.to_tensor(self._scene_idxs)
        self._scene_idxs = self._scene_idxs.to(torch.int).to(self.device)

    def __str__(self):
        return f"<struct of type {self.__class__}; managing {self._num_objs} {self._objs[0].__class__} objects>"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return self.__maniskill_hash__

    @cached_property
    def __maniskill_hash__(self):
        """A better hash to use compared to the default frozen dataclass hash.
        It is tied directly to the only immutable field (the _objs list)."""
        return hash(tuple([obj.__hash__() for obj in self._objs]))

    @property
    def device(self):
        return self.scene.device

    @property
    def _num_objs(self):
        return len(self._objs)

    @property
    def px(self):
        """The physx system objects managed by this dataclass are working on"""
        return self.scene.px


@dataclass
class PhysxRigidBaseComponentStruct(BaseStruct[T], Generic[T]):
    _bodies: list[physx.PhysxRigidBaseComponent]

    # ---------------------------------------------------------------------------- #
    # API from physx.PhysxRigidBaseComponent
    # ---------------------------------------------------------------------------- #
    # TODO (stao): To be added
    # def attach(self, collision_shape: PhysxCollisionShape) -> PhysxRigidBaseComponent:
    #     ...
    # def compute_global_aabb_tight(self) -> numpy.ndarray[tuple[typing.Literal[2], typing.Literal[3]], numpy.dtype[numpy.float32]]:
    #     ...
    # def get_collision_shapes(self) -> list[PhysxCollisionShape]:
    #     ...
    # def get_global_aabb_fast(self) -> numpy.ndarray[tuple[typing.Literal[2], typing.Literal[3]], numpy.dtype[numpy.float32]]:
    #     ...
    # @property
    # def _physx_pointer(self) -> int:
    #     ...
    # @property
    # def collision_shapes(self) -> list[PhysxCollisionShape]:
    #     ...


@dataclass
class PhysxRigidBodyComponentStruct(PhysxRigidBaseComponentStruct[T], Generic[T]):
    _bodies: list[physx.PhysxRigidBodyComponent]
    _body_data_name: str
    _body_data_index_internal: slice = None

    @property
    def px(self):
        """The physx system objects managed by this dataclass are working on"""
        return self.scene.px

    @cached_property
    def _body_data_index(self):
        """a list of indexes of each GPU rigid body in the `px.cuda_rigid_body_data` buffer, one for each element in `self._objs`"""
        if self._body_data_index_internal is None:
            self._body_data_index_internal = torch.tensor(
                [body.gpu_pose_index for body in self._bodies], device=self.device
            )
        return self._body_data_index_internal

    @property
    def _body_data(self) -> torch.Tensor:
        return getattr(self.px, self._body_data_name).torch()

    @cached_property
    def _body_force_query(self):
        return self.px.gpu_create_contact_body_impulse_query(self._bodies)

    def get_net_contact_forces(self):
        """
        Get the net contact forces on this body. Returns force vector of shape (N, 3)
        where N is the number of environments, and 3 is the dimension of the force vector itself,
        representing x, y, and z direction of force.
        """
        return self.get_net_contact_impulses() / self.scene.timestep

    def get_net_contact_impulses(self):
        """
        Get the net contact impulses on this body. Returns impulse vector of shape (N, 3)
        where N is the number of environments, and 3 is the dimension of the impulse vector itself,
        representing x, y, and z direction of impulse.
        """
        if self.scene.gpu_sim_enabled:
            self.px.gpu_query_contact_body_impulses(self._body_force_query)
            return self._body_force_query.cuda_impulses.torch().clone()
        else:
            body_contacts = sapien_utils.get_cpu_actor_contacts(
                self.px.get_contacts(), self._bodies[0].entity
            )
            net_force = common.to_tensor(
                sapien_utils.compute_total_impulse(body_contacts)
            )
            return net_force[None, :]

    # ---------------------------------------------------------------------------- #
    # API from physx.PhysxRigidBodyComponent
    # ---------------------------------------------------------------------------- #

    # TODO: To be added
    # def add_force_at_point(self, force: numpy.ndarray[numpy.float32, _Shape, _Shape[3]], point: numpy.ndarray[numpy.float32, _Shape, _Shape[3]], mode: typing.Literal['force', 'acceleration', 'velocity_change', 'impulse'] = 'force') -> None: ...
    # def add_force_torque(self, force: numpy.ndarray[numpy.float32, _Shape, _Shape[3]], torque: numpy.ndarray[numpy.float32, _Shape, _Shape[3]], mode: typing.Literal['force', 'acceleration', 'velocity_change', 'impulse'] = 'force') -> None: ...
    def get_angular_damping(self) -> float:
        return self.angular_damping

    def get_angular_velocity(self) -> torch.Tensor:
        return self.angular_velocity

    def get_auto_compute_mass(self) -> bool:
        return self.auto_compute_mass

    def get_cmass_local_pose(self) -> Pose:
        return

    def get_disable_gravity(self) -> bool:
        return self.disable_gravity

    # def get_inertia(self) -> numpy.ndarray[numpy.float32, _Shape, _Shape[3]]: ...
    def get_linear_damping(self) -> float:
        return self.linear_damping

    def get_linear_velocity(self) -> torch.Tensor:
        return self.linear_velocity

    def get_mass(self) -> float:
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
        if self.scene.gpu_sim_enabled:
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
        if self.scene.gpu_sim_enabled:
            # NOTE (stao): SAPIEN version 3.0.0b1 gpu sim has a bug inherited from physx where linear/angular velocities are in the wrong order
            # for link entities, namely 7:10 was angular velocity and 10:13 was linear velocity. SAPIEN 3.0.0 and above fixes this
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

    def get_gpu_index(self) -> int:
        return self.gpu_index

    def get_gpu_pose_index(self) -> int:
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
            arg0: The angular velocity to set. Can be of shape (N, 3) where N is the number of managed bodies or (3, ) to apply the same angular velocity to all managed bodies.
        """
        self.angular_velocity = arg0

    # def set_kinematic(self, arg0: bool) -> None: ...
    # def set_kinematic_target(self, arg0: sapien.pysapien.Pose) -> None: ...
    def set_linear_velocity(self, arg0: Array):
        """
        Set the linear velocity of the dynamic rigid body.
        Args:
            arg0: The linear velocity to set. Can be of shape (N, 3) where N is the number of managed bodies or (3, ) to apply the same linear velocity to all managed bodies.
        """
        self.linear_velocity = arg0

    def set_locked_motion_axes(self, axes: Array) -> None:
        """
        Set some motion axes of the dynamic rigid body to be locked
        Args:
            axes: list of 6 true/false values indicating whether which  of the 6 DOFs of the body is locked.
                  The order is linear X, Y, Z followed by angular X, Y, Z. If given a single list of length 6, it will be applied to all managed bodies.
                  If given a a batch of shape (N, 6), you can modify the N managed bodies each in batch.

        Example:
            set_locked_motion_axes([True, False, False, False, True, False]) allows the object to move along the X axis and rotate about the Y axis
        """
        self.locked_motion_axes = axes

    # def wake_up(self) -> None: ...
    @property
    def angular_velocity(self) -> torch.Tensor:
        if self.scene.gpu_sim_enabled:
            return self._body_data[self._body_data_index, 10:13]
        else:
            return torch.from_numpy(self._bodies[0].angular_velocity[None, :]).to(
                self.device
            )

    @angular_velocity.setter
    def angular_velocity(self, arg1: Array):
        if self.scene.gpu_sim_enabled:
            arg1 = common.to_tensor(arg1, device=self.device)
            self._body_data[
                self._body_data_index[self.scene._reset_mask[self._scene_idxs]], 10:13
            ] = arg1
        else:
            arg1 = common.to_numpy(arg1)
            if len(arg1.shape) == 2:
                arg1 = arg1[0]
            self._bodies[0].angular_velocity = arg1

    @property
    def gpu_index(self):
        if self.scene.gpu_sim_enabled:
            return [b.gpu_index for b in self._bodies]
        else:
            raise AttributeError("GPU index is not supported when gpu is not enabled")

    @property
    def gpu_pose_index(self):
        if self.scene.gpu_sim_enabled:
            return [b.gpu_pose_index for b in self._bodies]
        else:
            raise AttributeError(
                "GPU pose index is not supported when gpu is not enabled"
            )

    @property
    @before_gpu_init
    def is_sleeping(self):
        if self.scene.gpu_sim_enabled:
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
        if self.scene.gpu_sim_enabled:
            return self._body_data[self._body_data_index, 7:10]
        else:
            return torch.tensor(
                np.array([body.linear_velocity for body in self._bodies]),
                device=self.device,
            )

    @linear_velocity.setter
    def linear_velocity(self, arg1: Array):
        if self.scene.gpu_sim_enabled:
            arg1 = common.to_tensor(arg1, device=self.device)
            self._body_data[
                self._body_data_index[self.scene._reset_mask[self._scene_idxs]], 7:10
            ] = arg1
        else:
            arg1 = common.to_numpy(arg1)
            if len(arg1.shape) == 2:
                arg1 = arg1[0]
            self._bodies[0].linear_velocity = arg1

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
class PhysxJointComponentStruct(BaseStruct[T], Generic[T]):
    # def create(cls, bodies: Sequence[PhysxRigidBodyComponentStruct], parent_bodies: Sequence[PhysxRigidBodyComponentStruct]):
    # TODO
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
