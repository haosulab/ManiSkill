from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from mani_skill.sim.builders.base_builder import BaseBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Vec3

if TYPE_CHECKING:
    from mani_skill.sim.base_sim import BaseSim


class BaseActorBuilder(BaseBuilder, ABC):
    # TODO (stao): can this re-use for soft body? need to check newton api
    """Base actor builder for building rigid body objects (actors) in a simulation.
    Actor builders for each simulator backend should inherit from this class."""

    # NOTE (stao): most sims have a concept of a initial pose
    initial_pose: Pose | None = None
    """The initial pose of the actor when it gets built and spawned into the simulation."""

    scene_idxs: list[int] | None = None
    """The list of scene indices to build this actor in. If None, the actor will be
    built in all scenes."""

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

    @abstractmethod
    def build(self, name: str) -> Actor:
        """
        Build the actor.

        Args:
            name: The name of the actor.

        Returns:
            The built actor.
        """

    ### Standard primitive building functions, based on Sapien's original ActorBuilder ###
    def add_box_collision(
        self,
        pose: Pose,
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
        raise NotImplementedError("")

    def add_box_visual(
        self,
        pose: Pose,
        half_size: Vec3 = (1.0, 1.0, 1.0),
        material: Any | Vec3 | None = None,
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

        Returns:
            The actor builder.
        """
        raise NotImplementedError("")

    def add_sphere_collision(
        self,
        pose: Pose,
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
        raise NotImplementedError("")

    def add_sphere_visual(
        self,
        pose: Pose,
        radius: float = 1.0,
        material: Any | Vec3 | None = None,
    ) -> "BaseActorBuilder":
        """
        Add a sphere visual to the actor.

        Args:
            pose: The pose of the sphere relative to actor's local frame.
            radius: The radius of the sphere.
            material: The material of the sphere. This is dependent on simulator backend used.
            For SAPIEN this is a `sapien.render.RenderMaterial` object.
            For Newton based backends this is a ShapeCfg object.

        Returns:
            The actor builder.
        """
        raise NotImplementedError("")


class ActorBuilder(BaseActorBuilder):
    """Actor builder for building rigid body objects (actors) in a simulation. This
    is simulator independent and can be used to build actors across different simulators
    simultaneously to support e.g. rendering in one simulator and running physics in another."""

    _sims: dict[str, BaseSim] = {}
    """dictionary of simulators that will be tracking this builder. There can be multiple simulators
    that track this builder in order to support using different simulators for physics and
    rendering."""

    _sim_builders: dict[str, BaseActorBuilder] = {}
    """dictionary mapping sim id to the corresponding actor builder for that simulator."""

    def __init__(self):
        pass

    def add_sim(self, sim: BaseSim):
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

    def remove_sim(self, sim: BaseSim):
        """
        Remove a simulation backend that is tracking this builder.

        Args:
            sim: The simulation backend to remove.

        Returns:
            The actor builder.
        """
        self._sims.pop(sim.id)
        return self

    def build(self):
        pass

    def add_box_collision(
        self,
        pose: Pose,
        half_size: Vec3 = (1.0, 1.0, 1.0),
        material: Any | Vec3 | None = None,
        density: float = 1000.0,
    ) -> "ActorBuilder":
        for sim in self._sims.values():
            self._sim_builders[sim.id].add_box_collision(
                pose=pose,
                half_size=half_size,
                material=material,
                density=density,
            )
        return self
