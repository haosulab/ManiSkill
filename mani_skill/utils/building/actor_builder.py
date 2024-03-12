from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import numpy as np
import sapien
import sapien.physx as physx
import torch
from sapien import ActorBuilder as SAPIENActorBuilder
from sapien.wrapper.coacd import do_coacd

from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose, to_sapien_pose

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene


class ActorBuilder(SAPIENActorBuilder):
    """
    ActorBuilder class to flexibly build actors in both CPU and GPU simulations.
    This directly inherits the original flexible ActorBuilder from sapien and changes the build functions to support a batch of scenes and return a batch of Actors
    """

    scene: ManiSkillScene

    def __init__(self):
        super().__init__()
        self.initial_pose = Pose.create(self.initial_pose)
        self.scene_idxs = None
        self._allow_overlapping_plane_collisions = False
        self._plane_collision_poses = set()

    def set_scene_idxs(
        self,
        scene_idxs: Optional[
            Union[List[int], Sequence[int], torch.Tensor, np.ndarray]
        ] = None,
    ):
        """
        Set a list of scene indices to build this object in. Cannot be used in conjunction with scene mask
        """
        self.scene_idxs = scene_idxs
        return self

    def set_allow_overlapping_plane_collisions(self, v: bool):
        """Set whether or not to permit allowing overlapping plane collisions. In general if you are creating an Actor with a plane collision that is parallelized across multiple
        sub-scenes, you only need one of those collision shapes. If you add multiple, it will cause the simulation to slow down significantly. By default this is set to False"""
        self._allow_overlapping_plane_collisions = v
        return self

    def build_physx_component(self, link_parent=None):
        for r in self.collision_records:
            assert isinstance(r.material, sapien.physx.PhysxMaterial)

        if self.physx_body_type == "dynamic":
            component = sapien.physx.PhysxRigidDynamicComponent()
        elif self.physx_body_type == "kinematic":
            component = sapien.physx.PhysxRigidDynamicComponent()
            component.kinematic = True
        elif self.physx_body_type == "static":
            component = sapien.physx.PhysxRigidStaticComponent()
        elif self.physx_body_type == "link":
            component = sapien.physx.PhysxArticulationLinkComponent(link_parent)
        else:
            raise Exception(f"invalid physx body type [{self.physx_body_type}]")

        for r in self.collision_records:
            try:
                if r.type == "plane":
                    # skip adding plane collisions if we already added one.
                    pose_key = (tuple(r.pose.p), tuple(r.pose.q))
                    if (
                        self._allow_overlapping_plane_collisions
                        or pose_key not in self._plane_collision_poses
                    ):
                        shape = sapien.physx.PhysxCollisionShapePlane(
                            material=r.material,
                        )
                        shapes = [shape]
                        self._plane_collision_poses.add(pose_key)
                    else:
                        continue
                elif r.type == "box":
                    shape = sapien.physx.PhysxCollisionShapeBox(
                        half_size=r.scale, material=r.material
                    )
                    shapes = [shape]
                elif r.type == "capsule":
                    shape = sapien.physx.PhysxCollisionShapeCapsule(
                        radius=r.radius,
                        half_length=r.length,
                        material=r.material,
                    )
                    shapes = [shape]
                elif r.type == "cylinder":
                    shape = sapien.physx.PhysxCollisionShapeCylinder(
                        radius=r.radius,
                        half_length=r.length,
                        material=r.material,
                    )
                    shapes = [shape]
                elif r.type == "sphere":
                    shape = sapien.physx.PhysxCollisionShapeSphere(
                        radius=r.radius,
                        material=r.material,
                    )
                    shapes = [shape]
                elif r.type == "convex_mesh":
                    shape = sapien.physx.PhysxCollisionShapeConvexMesh(
                        filename=r.filename,
                        scale=r.scale,
                        material=r.material,
                    )
                    shapes = [shape]
                elif r.type == "nonconvex_mesh":
                    shape = sapien.physx.PhysxCollisionShapeTriangleMesh(
                        filename=r.filename,
                        scale=r.scale,
                        material=r.material,
                    )
                    shapes = [shape]
                elif r.type == "multiple_convex_meshes":
                    if r.decomposition == "coacd":
                        params = r.decomposition_params
                        if params is None:
                            params = dict()

                        filename = do_coacd(r.filename, **params)
                    else:
                        filename = r.filename

                    shapes = sapien.physx.PhysxCollisionShapeConvexMesh.load_multiple(
                        filename=filename,
                        scale=r.scale,
                        material=r.material,
                    )
                else:
                    raise RuntimeError(f"invalid collision shape type [{r.type}]")
            except RuntimeError:
                # ignore runtime error (e.g., failed to cooke mesh)
                continue

            for shape in shapes:
                shape.local_pose = r.pose
                shape.set_collision_groups(self.collision_groups)
                shape.set_density(r.density)
                shape.set_patch_radius(r.patch_radius)
                shape.set_min_patch_radius(r.min_patch_radius)
                component.attach(shape)

        if not self._auto_inertial and self.physx_body_type != "kinematic":
            component.mass = self._mass
            component.cmass_local_pose = self._cmass_local_pose
            component.inertia = self._inertia

        return component

    def build_kinematic(self, name):
        self.set_physx_body_type("kinematic")
        return self.build(name=name)

    def build_static(self, name):
        self.set_physx_body_type("static")
        return self.build(name=name)

    def build(self, name):
        """
        Build the actor with the given name.

        Different to the original SAPIEN API, a unique name is required here.
        """
        self.set_name(name)

        num_actors = self.scene.num_envs
        if self.scene_idxs is not None:
            pass
        else:
            self.scene_idxs = torch.arange((self.scene.num_envs), dtype=int)
        num_actors = len(self.scene_idxs)
        initial_pose = Pose.create(self.initial_pose)
        initial_pose_b = initial_pose.raw_pose.shape[0]
        assert initial_pose_b == 1 or initial_pose_b == num_actors
        initial_pose_np = sapien_utils.to_numpy(initial_pose.raw_pose)

        entities = []
        i = 0
        for scene_idx in self.scene_idxs:
            sub_scene = self.scene.sub_scenes[scene_idx]
            entity = self.build_entity()
            # prepend scene idx to entity name to indicate which sub-scene it is in
            entity.name = f"scene-{scene_idx}_{self.name}"
            # set pose before adding to scene
            if initial_pose_b == 1:
                entity.pose = to_sapien_pose(initial_pose_np)
            else:
                entity.pose = to_sapien_pose(initial_pose_np[i])
            sub_scene.add_entity(entity)
            entities.append(entity)
            i += 1
        actor = Actor.create_from_entities(entities, self.scene, self.scene_idxs)

        # if it is a static body type and this is a GPU sim but we are given a single initial pose, we repeat it for the purposes of observations
        if (
            self.physx_body_type == "static"
            and initial_pose_b == 1
            and physx.is_gpu_enabled()
        ):
            actor._builder_initial_pose = Pose.create(
                initial_pose.raw_pose.repeat(num_actors, 1)
            )
        else:
            actor._builder_initial_pose = initial_pose
        self.scene.actors[self.name] = actor
        return actor
