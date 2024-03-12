from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Generic, List, TypeVar

import sapien.physx as physx
import torch

from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.decorators import before_gpu_init
from mani_skill.utils.structs.types import Array

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
T = TypeVar("T")


@dataclass
class BaseStruct(Generic[T]):
    """
    Base class of all structs that manage sapien objects on CPU/GPU
    """

    _objs: List[T]
    """list of objects of type T managed by this dataclass"""
    _scene_idxs: torch.Tensor
    """parallel list with _objs indicating which sub-scene each of those objects are actually in by index"""
    _scene: ManiSkillScene
    """The ManiSkillScene object that manages the sub-scenes this dataclasses's objects are in"""

    def __post_init__(self):
        if not isinstance(self._scene_idxs, torch.Tensor):
            self._scene_idxs = sapien_utils.to_tensor(self._scene_idxs)

    @property
    def device(self):
        if physx.is_gpu_enabled():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    @property
    def _num_objs(self):
        return len(self._objs)

    @property
    def px(self):
        """The physx system objects managed by this dataclass are working on"""
        return self._scene.px


@dataclass
class PhysxRigidBodyComponentStruct:
    # Reference to the data for this rigid body on the GPU
    _scene: ManiSkillScene
    """The ManiSkillScene object that manages the sub-scenes this dataclasses's objects are in"""
    _body_data_name: str
    _bodies: List[physx.PhysxRigidBodyComponent]
    _body_data_index_internal: slice = None

    @property
    def px(self):
        """The physx system objects managed by this dataclass are working on"""
        return self._scene.px

    @cached_property
    def _body_data_index(self):
        if self._body_data_index_internal is None:
            self._body_data_index_internal = torch.tensor(
                [body.gpu_pose_index for body in self._bodies], device="cuda"
            )
        return self._body_data_index_internal

    @property
    def _body_data(self) -> torch.Tensor:
        return getattr(self.px, self._body_data_name).torch()

    @cached_property
    def _body_force_query(self):
        return self.px.gpu_create_contact_body_impulse_query(self._bodies)

    def get_net_contact_forces(self):
        if physx.is_gpu_enabled():
            self.px.gpu_query_contact_body_impulses(self._body_force_query)
            # NOTE (stao): physx5 calls the output forces but they are actually impulses
            return (
                self._body_force_query.cuda_impulses.torch().clone()
                / self._scene.timestep
            )
        else:
            body_contacts = sapien_utils.get_actor_contacts(
                self.px.get_contacts(), self._bodies[0].entity
            )
            net_force = (
                sapien_utils.to_tensor(
                    sapien_utils.compute_total_impulse(body_contacts)
                )
                / self._scene.timestep
            )
            return net_force[None, :]

    # ---------------------------------------------------------------------------- #
    # API from physx.PhysxRigidBodyComponent
    # ---------------------------------------------------------------------------- #

    # def add_force_at_point(self, force: numpy.ndarray[numpy.float32, _Shape, _Shape[3]], point: numpy.ndarray[numpy.float32, _Shape, _Shape[3]], mode: typing.Literal['force', 'acceleration', 'velocity_change', 'impulse'] = 'force') -> None: ...
    # def add_force_torque(self, force: numpy.ndarray[numpy.float32, _Shape, _Shape[3]], torque: numpy.ndarray[numpy.float32, _Shape, _Shape[3]], mode: typing.Literal['force', 'acceleration', 'velocity_change', 'impulse'] = 'force') -> None: ...
    def get_angular_damping(self) -> float:
        return self.angular_damping

    def get_angular_velocity(self) -> torch.Tensor:
        return self.angular_velocity

    def get_auto_compute_mass(self) -> bool:
        return self.auto_compute_mass

    # def get_cmass_local_pose(self) -> sapien.pysapien.Pose: ...
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
        if physx.is_gpu_enabled():
            return self._body_data[self._body_data_index, 10:13]
        else:
            return torch.from_numpy(self._bodies[0].angular_velocity[None, :])

    @property
    def auto_compute_mass(self) -> torch.Tensor:
        return torch.tensor([body.auto_compute_mass for body in self._bodies])

    # @property
    # def cmass_local_pose(self) -> sapien.pysapien.Pose:
    #     """
    #     :type: sapien.pysapien.Pose
    #     """
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
        if physx.is_gpu_enabled():
            return self._body_data[self._body_data_index, 7:10]
        else:
            return torch.from_numpy(self._bodies[0].linear_velocity[None, :])

    @property
    def mass(self) -> torch.Tensor:
        return torch.tensor([body.mass for body in self._bodies])

    @mass.setter
    @before_gpu_init
    def mass(self, arg1: float) -> None:
        if physx.is_gpu_enabled():
            raise NotImplementedError(
                "Setting mass is not supported on GPU sim at the moment."
            )
        else:
            self._bodies[0].mass = arg1

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
class PhysxRigidDynamicComponentStruct(PhysxRigidBodyComponentStruct):
    _bodies: List[physx.PhysxRigidDynamicComponent]

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
    # def get_locked_motion_axes(self) -> list[bool]: ...

    # def put_to_sleep(self) -> None: ...
    def set_angular_velocity(self, arg0: Array):
        self.angular_velocity = arg0

    # def set_kinematic(self, arg0: bool) -> None: ...
    # def set_kinematic_target(self, arg0: sapien.pysapien.Pose) -> None: ...
    def set_linear_velocity(self, arg0: Array):
        self.linear_velocity = arg0

    # def set_locked_motion_axes(self, axes: list[bool]) -> None:
    #     """
    #     set some motion axes of the dynamic rigid body to be locked
    #     Args:
    #         axes: list of 6 true/false values indicating whether which  of the 6 DOFs of the body is locked.
    #               The order is linear X, Y, Z followed by angular X, Y, Z.

    #     Example:
    #         set_locked_motion_axes([True, False, False, False, True, False]) allows the object to move along the X axis and rotate about the Y axis
    #     """
    # def wake_up(self) -> None: ...
    @property
    def angular_velocity(self) -> torch.Tensor:
        if physx.is_gpu_enabled():
            # TODO (stao): turn 10:13 etc. slices into constants
            return self._body_data[self._body_data_index, 10:13]
        else:
            return torch.from_numpy(self._bodies[0].angular_velocity[None, :])

    @angular_velocity.setter
    def angular_velocity(self, arg1: Array):
        if physx.is_gpu_enabled():
            arg1 = sapien_utils.to_tensor(arg1)
            self._body_data[
                self._body_data_index[self._scene._reset_mask], 10:13
            ] = arg1
        else:
            arg1 = sapien_utils.to_numpy(arg1)
            if len(arg1.shape) == 2:
                arg1 = arg1[0]
            self._bodies[0].angular_velocity = arg1

    @property
    def gpu_index(self):
        if physx.is_gpu_enabled():
            return [b.gpu_index for b in self._bodies]
        else:
            raise AttributeError("GPU index is not supported when gpu is not enabled")

    @property
    def gpu_pose_index(self):
        if physx.is_gpu_enabled():
            return [b.gpu_pose_index for b in self._bodies]
        else:
            raise AttributeError(
                "GPU pose index is not supported when gpu is not enabled"
            )

    @property
    @before_gpu_init
    def is_sleeping(self):
        if physx.is_gpu_enabled():
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
        if physx.is_gpu_enabled():
            return self._body_data[self._body_data_index, 7:10]
        else:
            return torch.from_numpy(self._bodies[0].linear_velocity[None, :])

    @linear_velocity.setter
    def linear_velocity(self, arg1: Array):
        if physx.is_gpu_enabled():
            arg1 = sapien_utils.to_tensor(arg1)
            self._body_data[self._body_data_index[self._scene._reset_mask], 7:10] = arg1
        else:
            arg1 = sapien_utils.to_numpy(arg1)
            if len(arg1.shape) == 2:
                arg1 = arg1[0]
            self._bodies[0].linear_velocity = arg1

    # @property
    # def locked_motion_axes(self) -> list[bool]:
    #     """
    #     :type: list[bool]
    #     """
