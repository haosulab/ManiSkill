from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sapien

from mani_skill.sim.builders.base_builder import BaseBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Vec3

if TYPE_CHECKING:
    from mani_skill.sim.base_sim import BaseSim


class BaseActorBuilder(BaseBuilder):
    # TODO (stao): can this re-use for soft body? need to check newton api
    """Base actor builder for building rigid body objects (actors) in a simulation.
    Actor builders for each simulator backend should inherit from this class.

    Not all building functions need to be implemented. Errors are thrown
    if un-implemented functions are called.
    """

    initial_pose: Pose | None = None
    """The initial pose of the actor when it gets built and spawned into the simulation."""

    scene_idxs: list[int] | None = None
    """The list of scene indices to build this actor in. If None, the actor will be
    built in all scenes."""

    _sims: dict[str, BaseSim] = {}
    """Dictionary of simulators that will be tracking this builder. There can be multiple simulators
    that track this builder in order to support using different simulators for physics and
    rendering. If you are writing an ActorBuilder for a simulator package (e.g. Sapien/Newton),
    you should not be accessing this attribute ever."""

    _sim_builders: dict[str, BaseActorBuilder] = {}
    """Dictionary mapping sim id to the corresponding actor builder for that simulator."""

    # NOTE (stao): To reduce the amount of duplicate code, we check if calls
    # calls to actor builders are from a concrete ActorBuilder (for e.g. Sapien/Newton) or
    # from the BaseActorBuilder. If the call is from the BaseActorBuilder, we proceed
    # If not and _sims is accessed, this indicates that the concrete ActorBuilder did not
    # implement the particular modular building function (e.g. add_box_collision).
    # this design makes it so that we can have a shared BaseActorBuilder for users
    # to build actors across different simulators while on the backend we can implement
    # a subset of the full actor builder functionality and don't need  two BaseActorBuilder
    # classes.

    def _add_sim(self, sim: BaseSim):
        """
        Add a simulation backend that should track this builder. Whenever this actor is built,
        the simulator backend will include this actor in its state and compile it in the scene.

        Args:
            sim: The simulation backend to add.

        Returns:
            The actor builder.
        """
        self._sims[sim.id] = sim
        self._sim_builders[sim.id] = sim.create_actor_builder()
        return self

    def _remove_sim(self, sim: BaseSim):
        """
        Remove a simulation backend that is tracking this builder.

        Args:
            sim: The simulation backend to remove.

        Returns:
            The actor builder.
        """
        self._sims.pop(sim.id)
        return self

    def set_initial_pose(self, initial_pose: Pose | None = None):
        """
        Set the initial pose of the actor. This is the pose of the actor when it is built and 
        spawned into the simulation before any physics steps are taken.
        """
        self.initial_pose = initial_pose
        return self

    def set_scene_idxs(self, scene_idxs: list[int] | None = None):
        """
        Set the sub-scene indices (parallel environment IDs) to build this actor in.

        Args:
            scene_idxs: The list of scene indices to build this actor in. If None, the
            actor will be built in all sub-scenes.

        Returns:
            The actor builder.
        """
        self.scene_idxs = scene_idxs
        return self

    def build(self, name: str) -> Actor:
        """
        Build the actor.

        Args:
            name: The name of the actor.

        Returns:
            The built actor.
        """
        for sim in self._sims.values():
            if type(self._sim_builders[sim.id]) is BaseActorBuilder:
                raise NotImplementedError(
                    f"{self._sim_builders[sim.id].__class__.__name__} does not support "
                    "build."
                )
            actor = self._sim_builders[sim.id].build(name=name)
        return actor

    def build_kinematic(self, name: str) -> Actor:
        """
        Build the actor as a kinematic object.

        Args:
            name: The name of the actor.

        Returns:
            The built actor.
        """
        for sim in self._sims.values():
            if type(self._sim_builders[sim.id]) is BaseActorBuilder:
                raise NotImplementedError(
                    f"{self._sim_builders[sim.id].__class__.__name__} does not support "
                    "build_kinematic."
                )
            actor = self._sim_builders[sim.id].build_kinematic(name=name)
        return actor

    ### Standard primitive building functions, based on Sapien's original ActorBuilder ###
    def add_plane_collision(
        self,
        pose: Pose | None = None,
        material: Any | None = None,
        patch_radius: float = 0,
        min_patch_radius: float = 0,
    ):
        for sim in self._sims.values():
            if type(self._sim_builders[sim.id]) is BaseActorBuilder:
                raise NotImplementedError(
                    f"{self._sim_builders[sim.id].__class__.__name__} does not support "
                    "add_plane_collision."
                )
            self._sim_builders[sim.id].add_plane_collision(
                pose=pose if pose is not None else sapien.Pose(),
                material=material,
                patch_radius=patch_radius,
                min_patch_radius=min_patch_radius,
            )
        return self

    def add_plane_visual(
        self,
        pose: Pose | None = None,
        scale: Vec3 = (1, 1, 1),
        material: Any | Vec3 | None = None,
        name: str = "",
    ):
        for sim in self._sims.values():
            if type(self._sim_builders[sim.id]) is BaseActorBuilder:
                raise NotImplementedError(
                    f"{self._sim_builders[sim.id].__class__.__name__} does not support "
                    "add_plane_visual."
                )
            self._sim_builders[sim.id].add_plane_visual(
                pose=pose if pose is not None else sapien.Pose(),
                scale=scale,
                material=material,
                name=name,
            )
        return self

    def add_capsule_collision(
        self,
        pose: Pose | None = None,
        radius: float = 1,
        half_length: float = 1,
        material: Any | None = None,
        density: float = 1000,
        patch_radius: float = 0,
        min_patch_radius: float = 0,
    ):
        for sim in self._sims.values():
            if type(self._sim_builders[sim.id]) is BaseActorBuilder:
                raise NotImplementedError(
                    f"{self._sim_builders[sim.id].__class__.__name__} does not support "
                    "add_capsule_collision."
                )
            self._sim_builders[sim.id].add_capsule_collision(
                pose=pose if pose is not None else sapien.Pose(),
                radius=radius,
                half_length=half_length,
                material=material,
                density=density,
                patch_radius=patch_radius,
                min_patch_radius=min_patch_radius,
            )
        return self
    
    def add_capsule_visual(
        self,
        pose: Pose | None = None,
        radius: float = 1,
        half_length: float = 1,
        material: Any | Vec3 | None = None,
        name: str = "",
    ):
        for sim in self._sims.values():
            if type(self._sim_builders[sim.id]) is BaseActorBuilder:
                raise NotImplementedError(
                    f"{self._sim_builders[sim.id].__class__.__name__} does not support "
                    "add_capsule_visual."
                )
            self._sim_builders[sim.id].add_capsule_visual(
                pose=pose if pose is not None else sapien.Pose(),
                radius=radius,
                half_length=half_length,
                material=material,
                name=name,
            )
        return self

    def add_cylinder_collision(
        self,
        pose: Pose | None = None,
        radius: float = 1,
        half_length: float = 1,
        material: Any | None = None,
        density: float = 1000,
        patch_radius: float = 0,
        min_patch_radius: float = 0,
    ):
        for sim in self._sims.values():
            if type(self._sim_builders[sim.id]) is BaseActorBuilder:
                raise NotImplementedError(
                    f"{self._sim_builders[sim.id].__class__.__name__} does not support "
                    "add_cylinder_collision."
                )
            self._sim_builders[sim.id].add_cylinder_collision(
                pose=pose if pose is not None else sapien.Pose(),
                radius=radius,
                half_length=half_length,
                material=material,
                density=density,
                patch_radius=patch_radius,
                min_patch_radius=min_patch_radius,
            )
        return self

    def add_cylinder_visual(
        self,
        pose: Pose | None = None,
        radius: float = 1,
        half_length: float = 1,
        material: Any | Vec3 | None = None,
        name: str = "",
    ):
        for sim in self._sims.values():
            if type(self._sim_builders[sim.id]) is BaseActorBuilder:
                raise NotImplementedError(
                    f"{self._sim_builders[sim.id].__class__.__name__} does not support "
                    "add_cylinder_visual."
                )
            self._sim_builders[sim.id].add_cylinder_visual(
                pose=pose if pose is not None else sapien.Pose(),
                radius=radius,
                half_length=half_length,
                material=material,
                name=name,
            )
        return self

    def add_box_collision(
        self,
        pose: Pose | None = None,
        half_size: Vec3 = (1.0, 1.0, 1.0),
        material: Any | None = None,
        density: float = 1000.0,
    ) -> "BaseActorBuilder":
        # TODO (stao): check Newton API here w.r.t concept of material config
        """
        Add a box collision to the actor.

        Args:
            pose: The pose of the box relative to actor's local frame.
            half_size: The half size of the box (x, y, and z dimensions).
            material: The material of the box. This is dependent on simulator backend used.
            For SAPIEN this is a `sapien.physx.PhysxMaterial` object.
            For Newton based backends this is a ShapeCfg object.
            density: The density of the box.

        Returns:
            The actor builder.
        """
        for sim in self._sims.values():
            if type(self._sim_builders[sim.id]) is BaseActorBuilder:
                raise NotImplementedError(
                    f"{self._sim_builders[sim.id].__class__.__name__} does not support "
                    "add_box_collision."
                )
            self._sim_builders[sim.id].add_box_collision(
                pose=pose if pose is not None else sapien.Pose(),
                half_size=half_size,
                material=material,
                density=density,
            )
        return self

    def add_box_visual(
        self,
        pose: Pose | None = None,
        half_size: Vec3 = (1.0, 1.0, 1.0),
        material: Any | Vec3 | None = None,
        name: str = "",
    ) -> "BaseActorBuilder":
        # TODO (stao): check Newton API here w.r.t concept of material config
        """
        Add a box visual to the actor.

        Args:
            pose: The pose of the box relative to actor's local frame.
            half_size: The half size of the box (x, y, and z dimensions).
            material: The material of the box. This is dependent on simulator backend used.
            For SAPIEN this is a `sapien.render.RenderMaterial` object.
            For Newton based backends this is a ShapeCfg object.
            name: The name of the box visual.

        Returns:
            The actor builder.
        """
        for sim in self._sims.values():
            if type(self._sim_builders[sim.id]) is BaseActorBuilder:
                raise NotImplementedError(
                    f"{self._sim_builders[sim.id].__class__.__name__} does not support "
                    "add_box_visual."
                )
            self._sim_builders[sim.id].add_box_visual(
                pose=pose if pose is not None else sapien.Pose(),
                half_size=half_size,
                material=material,
                name=name,
            )
        return self

    def add_sphere_collision(
        self,
        pose: Pose | None = None,
        radius: float = 1.0,
        material: Any | Vec3 | None = None,
        density: float = 1000.0,
    ) -> "BaseActorBuilder":
        """
        Add a sphere collision to the actor.

        Args:
            pose: The pose of the sphere relative to actor's local frame.
            radius: The radius of the sphere.
            material: The material of the sphere. This is dependent on simulator backend used.
            For SAPIEN this is a `sapien.physx.PhysxMaterial` object.
            For Newton based backends this is a ShapeCfg object.
            density: The density of the sphere.

        Returns:
            The actor builder.
        """
        for sim in self._sims.values():
            if type(self._sim_builders[sim.id]) is BaseActorBuilder:
                raise NotImplementedError(
                    f"{self._sim_builders[sim.id].__class__.__name__} does not support "
                    "add_sphere_collision."
                )
            self._sim_builders[sim.id].add_sphere_collision(
                pose=pose if pose is not None else sapien.Pose(),
                radius=radius,
                material=material,
                density=density,
            )
        return self

    def add_sphere_visual(
        self,
        pose: Pose,
        radius: float = 1.0,
        material: Any | Vec3 | None = None,
        name: str = "",
    ) -> "BaseActorBuilder":
        """
        Add a sphere visual to the actor.

        Args:
            pose: The pose of the sphere relative to actor's local frame.
            radius: The radius of the sphere.
            material: The material of the sphere. This is dependent on simulator backend used.
            For SAPIEN this is a `sapien.render.RenderMaterial` object.
            For Newton based backends this is a ShapeCfg object.
            name: The name of the sphere visual.

        Returns:
            The actor builder.
        """
        for sim in self._sims.values():
            if type(self._sim_builders[sim.id]) is BaseActorBuilder:
                raise NotImplementedError(
                    f"{self._sim_builders[sim.id].__class__.__name__} does not support "
                    "add_sphere_visual."
                )
            self._sim_builders[sim.id].add_sphere_visual(
                pose=pose if pose is not None else sapien.Pose(),
                radius=radius,
                material=material,
                name=name,
            )
        return self

    # def add_convex_collision_from_file(
    #     self,
    #     filename,
    #     pose: sapien.Pose = sapien.Pose(),
    #     scale: Vec3 = (1, 1, 1),
    #     material: Union[sapien.physx.PhysxMaterial, None] = None,
    #     density: float = 1000,
    #     patch_radius: float = 0,
    #     min_patch_radius: float = 0,
    #     is_trigger: bool = False,
    # ):
    #     """
    #     Add a convex collision from a file to the actor.
    #     """
    #     raise NotImplementedError("")

    # def add_multiple_convex_collisions_from_file(
    #     self,
    #     filename,
    #     pose: sapien.Pose = sapien.Pose(),
    #     scale: Vec3 = (1, 1, 1),
    #     material: Union[sapien.physx.PhysxMaterial, None] = None,
    #     density: float = 1000,
    #     patch_radius: float = 0,
    #     min_patch_radius: float = 0,
    #     is_trigger: bool = False,
    #     decomposition: typing.Literal["none", "coacd"] = "none",
    #     decomposition_params=dict(),
    # ):
    #     if material is None:
    #         material = sapien.physx.get_default_material()

    #     self.collision_records.append(
    #         CollisionShapeRecord(
    #             type="multiple_convex_meshes",
    #             filename=filename,
    #             pose=pose,
    #             scale=scale,
    #             material=material,
    #             density=density,
    #             patch_radius=patch_radius,
    #             min_patch_radius=min_patch_radius,
    #             is_trigger=is_trigger,
    #             decomposition=decomposition,
    #             decomposition_params=decomposition_params,
    #         )
    #     )
    #     return self

    # def add_nonconvex_collision_from_file(
    #     self,
    #     filename: str,
    #     pose: sapien.Pose = sapien.Pose(),
    #     scale: Vec3 = (1, 1, 1),
    #     material: Union[sapien.physx.PhysxMaterial, None] = None,
    #     density: float = 1000,
    #     patch_radius: float = 0,
    #     min_patch_radius: float = 0,
    #     is_trigger: bool = False,
    # ):
    #     if material is None:
    #         material = sapien.physx.get_default_material()

    #     self.collision_records.append(
    #         CollisionShapeRecord(
    #             type="nonconvex_mesh",
    #             filename=filename,
    #             pose=pose,
    #             scale=scale,
    #             material=material,
    #             density=density,
    #             patch_radius=patch_radius,
    #             min_patch_radius=min_patch_radius,
    #             is_trigger=is_trigger,
    #         )
    #     )
    #     return self

    def add_visual_from_file(
        self,
        filename: str,
        pose: Pose | None = None,
        scale: Vec3 = (1, 1, 1),
        material: Any | None = None,
        name: str = "",
    ):
        """
        Add a visual mesh from a file to the actor.

        Args:
            filename: The path to the file containing the visual mesh.
            pose: The pose of the visual mesh relative to actor's local frame.
            scale: The scale of the visual mesh.
            material: The material of the visual mesh. This is dependent on simulator backend used.
            name: The name of the visual mesh.

        Returns:
            The actor builder.
        """
        for sim in self._sims.values():
            if type(self._sim_builders[sim.id]) is BaseActorBuilder:
                raise NotImplementedError(
                    f"{self._sim_builders[sim.id].__class__.__name__} does not support "
                    "add_visual_from_file."
                )
            self._sim_builders[sim.id].add_visual_from_file(
                filename=filename,
                pose=pose,
                scale=scale,
                material=material,
                name=name,
            )
        return self
