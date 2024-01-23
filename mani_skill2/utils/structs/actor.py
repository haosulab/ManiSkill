from dataclasses import dataclass
from typing import List, Literal, Union

import numpy as np
import sapien
import sapien.physx as physx
import sapien.render
import torch

from mani_skill2.utils.sapien_utils import to_numpy
from mani_skill2.utils.structs.base import BaseStruct, PhysxRigidDynamicComponentStruct
from mani_skill2.utils.structs.pose import Pose, to_sapien_pose, vectorize_pose
from mani_skill2.utils.structs.types import Array


@dataclass
class Actor(PhysxRigidDynamicComponentStruct, BaseStruct[sapien.Entity]):
    """
    Wrapper around sapien.Entity objects mixed in with useful properties from the RigidBodyDynamicComponent components

    At the moment, on GPU and CPU one can query pose, linear velocity, and angular velocity easily

    On CPU, more properties are available
    """

    px_body_type: Literal["kinematic", "static", "dynamic"]
    _data_index: slice = None
    hidden: bool = False

    # track the initial pose of the actor builder for this actor. Necessary to ensure the actor is reset correctly once
    # gpu system is initialized
    _builder_initial_pose: sapien.Pose = None
    name: str = None

    @classmethod
    def create_from_entities(cls, entities: List[sapien.Entity]):
        px: Union[physx.PhysxSystem, physx.PhysxGpuSystem] = entities[
            0
        ].scene.physx_system
        shared_name = "_".join(entities[0].name.split("_")[1:])
        bodies = [
            ent.find_component_by_type(physx.PhysxRigidDynamicComponent)
            for ent in entities
        ]
        # Objects with collision shapes have either PhysxRigidDynamicComponent (Kinematic, Dynamic) or PhysxRigidStaticComponent (Static)
        px_body_type = "static"
        if bodies[0] is not None:
            if bodies[0].kinematic:
                px_body_type = "kinematic"
            else:
                px_body_type = "dynamic"
        return cls(
            _objs=entities,
            px=px,
            px_body_type=px_body_type,
            _bodies=bodies,
            _body_data_index=None,
            _body_data_name="cuda_rigid_body_data"
            if isinstance(px, physx.PhysxGpuSystem)
            else None,
            name=shared_name,
        )

    # -------------------------------------------------------------------------- #
    # Additional useful functions not in SAPIEN original API
    # -------------------------------------------------------------------------- #

    def get_state(self):
        pose = self.pose
        if self.px_body_type != "dynamic":
            vel = torch.zeros((self._num_objs, 3), device=self.device)
            ang_vel = torch.zeros((self._num_objs, 3), device=self.device)
        else:
            vel = self.get_linear_velocity()  # [N, 3]
            ang_vel = self.get_angular_velocity()  # [N, 3]
        return torch.hstack([pose.p, pose.q, vel, ang_vel])

    def set_state(self, state: Array):
        if physx.is_gpu_enabled():
            raise NotImplementedError(
                "You should not set state on a GPU enabled actor."
            )
        else:
            state = to_numpy(state[0])
            self.set_pose(sapien.Pose(state[0:3], state[3:7]))
            if self.px_body_type == "dynamic":
                self.set_linear_velocity(state[7:10])
                self.set_angular_velocity(state[10:13])

    def has_collision_shapes(self):
        return (
            len(
                self._objs[0]
                .find_component_by_type(physx.PhysxRigidDynamicComponent)
                .collision_shapes
            )
            > 0
        )

    def hide_visual(self):
        """
        Hides this actor from view. In CPU simulation the visual body is simply set to visibility 0

        For GPU simulation, currently this is implemented by moving the actor very far away as visiblity cannot be changed on the fly.
        As a result we do not permit hiding and showing visuals of objects with collision shapes as this affects the actual simulation
        """
        assert not self.has_collision_shapes()
        if self.hidden:
            return
        if physx.is_gpu_enabled():
            self.last_pose = self.px.cuda_rigid_body_data[
                self._body_data_index, :7
            ].clone()
            temp_pose = self.pose.raw_pose
            temp_pose[..., :3] += 99999
            self.pose = temp_pose
            self.px.gpu_apply_rigid_dynamic_data()
        else:
            self._objs[0].find_component_by_type(
                sapien.render.RenderBodyComponent
            ).visibility = 0
        self.hidden = True

    def show_visual(self):
        assert not self.has_collision_shapes()
        if not self.hidden:
            return
        if physx.is_gpu_enabled():
            if hasattr(self, "last_pose"):
                self.pose = self.last_pose
                self.px.gpu_apply_rigid_dynamic_data()
        else:
            self._objs[0].find_component_by_type(
                sapien.render.RenderBodyComponent
            ).visibility = 1
        self.hidden = False

    # -------------------------------------------------------------------------- #
    # Exposed actor properties, getters/setters that automatically handle
    # CPU and GPU based actors
    # -------------------------------------------------------------------------- #
    def remove_from_scene(self):
        if physx.is_gpu_enabled():
            raise RuntimeError(
                "Cannot physically remove object from scene during GPU simulation. This can only be done in CPU simulation. If you wish to remove an object physically, the best way is to move the object far away."
            )
        else:
            self._objs[0].remove_from_scene()

    @property
    def pose(self) -> Pose:
        if physx.is_gpu_enabled():
            if self.px_body_type == "static":
                # NOTE (stao): usually _builder_initial_pose is just one pose, but for static objects in GPU sim we repeat it if necessary so it can be used
                # as part of observations if needed
                return self._builder_initial_pose
            else:
                return Pose.create(
                    self.px.cuda_rigid_body_data[self._body_data_index, :7]
                )
        else:
            assert len(self._objs) == 1
            return Pose.create(self._objs[0].pose)

    @pose.setter
    def pose(self, arg1: Union[Pose, sapien.Pose, Array]) -> None:
        if physx.is_gpu_enabled():
            if not isinstance(arg1, torch.Tensor):
                arg1 = vectorize_pose(arg1)
            self.px.cuda_rigid_body_data[self._body_data_index, :7] = arg1
        else:
            self._objs[0].pose = to_sapien_pose(arg1)

    def set_pose(self, arg1: Union[Pose, sapien.Pose]) -> None:
        self.pose = arg1
