from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np
import sapien
import sapien.physx as physx
import torch
from sapien import ActorBuilder as SAPIENActorBuilder
from sapien.wrapper.coacd import do_coacd

from mani_skill import logger
from mani_skill.utils import common
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
        self.initial_pose = None
        self.scene_idxs = None
        self._allow_overlapping_plane_collisions = False
        self._plane_collision_poses = set()
        self._procedural_shapes = []
        """procedurally generated shapes to attach"""

    def set_scene_idxs(
        self,
        scene_idxs: Optional[
            Union[list[int], Sequence[int], torch.Tensor, np.ndarray]
        ] = None,
    ):
        """
        Set a list of scene indices to build this object in. Cannot be used in conjunction with scene mask
        """
        self.scene_idxs = scene_idxs
        return self

    def set_allow_overlapping_plane_collisions(self, v: bool):
        """Set whether or not to permit allowing overlapping plane collisions. In general if you are creating an Actor with a plane collision that is parallelized across multiple
        sub-scenes, you only need one of those collision shapes. If you add multiple, it will cause the simulation to slow down significantly. By default this is set to False
        """
        self._allow_overlapping_plane_collisions = v
        return self

    def build_physx_component(self, link_parent=None):
        for r in self.collision_records:
            assert isinstance(r.material, physx.PhysxMaterial)

        if self.physx_body_type == "dynamic":
            component = physx.PhysxRigidDynamicComponent()
        elif self.physx_body_type == "kinematic":
            component = physx.PhysxRigidDynamicComponent()
            component.kinematic = True
        elif self.physx_body_type == "static":
            component = physx.PhysxRigidStaticComponent()
        elif self.physx_body_type == "link":
            component = physx.PhysxArticulationLinkComponent(link_parent)
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
                        shape = physx.PhysxCollisionShapePlane(
                            material=r.material,
                        )
                        shapes = [shape]
                        self._plane_collision_poses.add(pose_key)
                    else:
                        continue
                elif r.type == "box":
                    shape = physx.PhysxCollisionShapeBox(
                        half_size=r.scale, material=r.material
                    )
                    shapes = [shape]
                elif r.type == "capsule":
                    shape = physx.PhysxCollisionShapeCapsule(
                        radius=r.radius,
                        half_length=r.length,
                        material=r.material,
                    )
                    shapes = [shape]
                elif r.type == "cylinder":
                    shape = physx.PhysxCollisionShapeCylinder(
                        radius=r.radius,
                        half_length=r.length,
                        material=r.material,
                    )
                    shapes = [shape]
                elif r.type == "sphere":
                    shape = physx.PhysxCollisionShapeSphere(
                        radius=r.radius,
                        material=r.material,
                    )
                    shapes = [shape]
                elif r.type == "convex_mesh":
                    shape = physx.PhysxCollisionShapeConvexMesh(
                        filename=r.filename,
                        scale=r.scale,
                        material=r.material,
                    )
                    shapes = [shape]
                elif r.type == "nonconvex_mesh":
                    shape = physx.PhysxCollisionShapeTriangleMesh(
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

                    shapes = physx.PhysxCollisionShapeConvexMesh.load_multiple(
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

        if hasattr(self, "_auto_inertial"):
            if not self._auto_inertial and self.physx_body_type != "kinematic":
                component.mass = self._mass
                component.cmass_local_pose = self._cmass_local_pose
                component.inertia = self._inertia

        component.name = self.name
        return component

    def build_dynamic(self, name):
        self.set_physx_body_type("dynamic")
        return self.build(name=name)

    def build_kinematic(self, name):
        self.set_physx_body_type("kinematic")
        return self.build(name=name)

    def build_static(self, name):
        self.set_physx_body_type("static")
        return self.build(name=name)

    def build_entity(self):
        """
        build the raw sapien entity. Modifies original SAPIEN function to accept new procedurally generated render components
        """
        entity = sapien.Entity()
        if self.scene.can_render():
            if self.visual_records or len(self._procedural_shapes) > 0:
                render_component = self.build_render_component()
                for shape in self._procedural_shapes:
                    render_component.attach(shape)
                entity.add_component(render_component)
        entity.add_component(self.build_physx_component())
        entity.name = self.name
        return entity

    def build(self, name):
        """
        Build the actor with the given name.

        Different to the original SAPIEN API, a unique name is required here.
        """
        self.set_name(name)

        assert (
            self.name is not None
            and self.name != ""
            and self.name not in self.scene.actors
        ), "built actors in ManiSkill must have unique names and cannot be None or empty strings"

        if self.scene_idxs is not None:
            self.scene_idxs = common.to_tensor(
                self.scene_idxs, device=self.scene.device
            ).to(torch.int)
        else:
            self.scene_idxs = torch.arange((self.scene.num_envs), dtype=int)
        num_actors = len(self.scene_idxs)

        if self.initial_pose is None:
            logger.warn(
                f"No initial pose set for actor builder of {self.name}, setting to default pose q=[1,0,0,0], p=[0,0,0]. Not setting reasonable initial poses may slow down simulation, see https://github.com/haosulab/ManiSkill/issues/421."
            )
            self.initial_pose = Pose.create(sapien.Pose())
        else:
            self.initial_pose = Pose.create(self.initial_pose, device=self.scene.device)

        initial_pose_b = self.initial_pose.raw_pose.shape[0]
        assert initial_pose_b == 1 or initial_pose_b == num_actors
        initial_pose_np = common.to_numpy(self.initial_pose.raw_pose)
        if initial_pose_b == 1:
            initial_pose_np = initial_pose_np.repeat(num_actors, axis=0)
        if self.scene.parallel_in_single_scene:
            initial_pose_np[:, :3] += self.scene.scene_offsets_np[
                common.to_numpy(self.scene_idxs)
            ]
        entities = []

        for i, scene_idx in enumerate(self.scene_idxs):
            if self.scene.parallel_in_single_scene:
                sub_scene = self.scene.sub_scenes[0]
            else:
                sub_scene = self.scene.sub_scenes[scene_idx]
            entity = self.build_entity()
            # prepend scene idx to entity name to indicate which sub-scene it is in
            entity.name = f"scene-{scene_idx}_{self.name}"
            # set pose before adding to scene
            entity.pose = to_sapien_pose(initial_pose_np[i])
            sub_scene.add_entity(entity)
            entities.append(entity)
        actor = Actor.create_from_entities(entities, self.scene, self.scene_idxs)

        # if it is a static body type and this is a GPU sim but we are given a single initial pose, we repeat it for the purposes of observations
        if (
            self.physx_body_type == "static"
            and initial_pose_b == 1
            and self.scene.gpu_sim_enabled
        ):
            actor.initial_pose = Pose.create(
                self.initial_pose.raw_pose.repeat(num_actors, 1)
            )
        else:
            actor.initial_pose = self.initial_pose
        self.scene.actors[self.name] = actor
        self.scene.add_to_state_dict_registry(actor)
        return actor

    """
    additional procedurally generated visual meshes
    """

    def add_plane_repeated_visual(
        self,
        pose: sapien.Pose = sapien.Pose(),
        half_size: list[float] = [5, 5],
        mat: sapien.render.RenderMaterial = None,
        texture_repeat: list[float] = [1, 1],
    ):
        """Procedurally generateds a repeated 2D texture. Works similarly to https://mujoco.readthedocs.io/en/stable/XMLreference.html#asset-material-texrepeat

        currently this always adds a back face

        Args:
            texture_repeat: the number of times to repeat the texture in each direction.
        """
        floor_width = half_size[0] * 2
        floor_length = half_size[1] * 2
        floor_width = int(np.ceil(floor_width))
        floor_length = int(np.ceil(floor_length))

        # generate a grid of right triangles that form 1x1 meter squares centered at (0, 0, 0)
        # for squares on the edge we cut them off

        # floor_length = floor_width if floor_length is None else floor_length
        num_verts = (floor_width + 1) * (floor_length + 1)
        vertices = np.zeros((int(num_verts), 3))
        floor_half_width = floor_width / 2
        floor_half_length = floor_length / 2
        xrange = np.arange(start=-floor_half_width, stop=floor_half_width + 1)
        yrange = np.arange(start=-floor_half_length, stop=floor_half_length + 1)
        xx, yy = np.meshgrid(xrange, yrange)
        xys = np.stack((xx, yy), axis=2).reshape(-1, 2)
        vertices[:, 0] = xys[:, 0]
        vertices[:, 1] = xys[:, 1]
        normals = np.zeros((len(vertices), 3))
        normals[:, 2] = -1

        # the number of times the texture repeats essentially.
        uvs = np.zeros((len(vertices), 2))
        # texture_repeat = [1,1]
        uvs[:, 0] = xys[:, 0] * texture_repeat[0]
        uvs[:, 1] = xys[:, 1] * texture_repeat[1]

        # TODO: This is fast but still two for loops which is a little annoying
        triangles = []
        for i in range(floor_length):
            triangles.append(
                np.stack(
                    [
                        np.arange(floor_width) + i * (floor_width + 1),
                        np.arange(floor_width)
                        + 1
                        + floor_width
                        + i * (floor_width + 1),
                        np.arange(floor_width) + 1 + i * (floor_width + 1),
                    ],
                    axis=1,
                )
            )
        for i in range(floor_length):
            triangles.append(
                np.stack(
                    [
                        np.arange(floor_width)
                        + 1
                        + floor_width
                        + i * (floor_width + 1),
                        np.arange(floor_width)
                        + floor_width
                        + 2
                        + i * (floor_width + 1),
                        np.arange(floor_width) + 1 + i * (floor_width + 1),
                    ],
                    axis=1,
                )
            )
        triangles = np.concatenate(triangles)

        # vertices: (N, 3)
        # triangles: (M, 3) of index triplets referencing vertices
        # normals: (N, 3) normals of the vertices. These should all face the same direction
        # uvs: (N, 2) uv coordinates for the vertices
        if half_size[0] < floor_half_width:
            diff = floor_half_width - half_size[0]
            for sign in [-1, 1]:
                mask = vertices[:, 0] == floor_half_width * sign
                vertices[mask, 0] = half_size[0] * sign
                uvs[mask, 0] -= 1 * diff * sign * texture_repeat[0]

        if half_size[1] < floor_half_length:
            diff = floor_half_length - half_size[1]
            for sign in [-1, 1]:
                mask = vertices[:, 1] == floor_half_length * sign
                vertices[mask, 1] = half_size[1] * sign
                uvs[mask, 1] -= diff * sign * texture_repeat[1]
        shape = sapien.render.RenderShapeTriangleMesh(
            vertices=vertices,
            triangles=triangles,
            normals=normals,
            uvs=uvs,
            material=mat,
        )
        shape.local_pose = pose
        self._procedural_shapes.append(shape)
