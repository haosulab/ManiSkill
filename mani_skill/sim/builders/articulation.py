from __future__ import annotations

from typing import TYPE_CHECKING

from mani_skill.sim.builders.base_builder import BaseBuilder
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.pose import Pose

if TYPE_CHECKING:
    from mani_skill.sim.base_sim import BaseSim


class BaseArticulationBuilder(BaseBuilder):
    """Articulation builder for building articulated objects in a simulation.
    This is simulator independent and can be used to build articulations across different simulators
    simultaneously to support e.g. rendering in one simulator and running physics in another."""

    scene_idxs: list[int] | None = None
    """The list of scene indices to build this articulation in. If None, the articulation will
    be built in all scenes"""

    __sims: dict[str, BaseSim] = {}
    """dictionary of simulators that will be tracking this builder. There can be multiple simulators
    that track this builder in order to support using different simulators for physics and
    rendering."""

    __sim_builders: dict[str, BaseArticulationBuilder] = {}
    """dictionary mapping sim id to the corresponding articulation builder for that simulator."""

    sim: BaseSim
    """The simulation backend this builder builds in"""

    def __init__(self):
        pass

    def _add_sim(self, sim: BaseSim):
        """
        Add a simulation backend that should track this builder. Whenever this articulation is
        built, the simulator backend will include this articulation in its state and compile
        it in the scene.

        Args:
            sim: The simulation backend to add.

        Returns:
            The articulation builder.
        """
        self.__sims[sim.id] = sim
        self.__sim_builders[sim.id] = sim.create_articulation_builder()
        return self

    def _remove_sim(self, sim: BaseSim):
        """
        Remove a simulation backend that is tracking this builder.
        """
        self.__sims.pop(sim.id)
        return self

    @property
    def initial_pose(self) -> Pose | None:
        """The initial pose of the actor when it gets built and spawned into the simulation."""
        return next(iter(self.__sim_builders.values())).initial_pose

    @initial_pose.setter
    def initial_pose(self, initial_pose: Pose | None):
        for sim in self.__sims.values():
            if type(self.__sim_builders[sim.id]) is BaseArticulationBuilder:
                raise NotImplementedError(
                    f"{self.__sim_builders[sim.id].__class__.__name__} does not support "
                    "initial_pose."
                )
            self.__sim_builders[sim.id].initial_pose = initial_pose

    def set_initial_pose(self, initial_pose: Pose | None = None):
        """
        Set the initial pose of the actor. This is the pose of the actor when it is built and
        spawned into the simulation before any physics steps are taken.
        """
        self.initial_pose = initial_pose

    def set_scene_idxs(self, scene_idxs: list[int] | None = None):
        """
        Set the sub-scene indices (parallel environment IDs) to build this actor in.

        Args:
            scene_idxs: The list of scene indices to build this actor in. If None, the
            actor will be built in all sub-scenes.

        Returns:
            The actor builder.
        """
        for sim in self.__sims.values():
            if type(self.__sim_builders[sim.id]) is BaseArticulationBuilder:
                raise NotImplementedError(
                    f"{self.__sim_builders[sim.id].__class__.__name__} does not support "
                    "set_scene_idxs."
                )
            self.__sim_builders[sim.id].set_scene_idxs(scene_idxs)

    def build(self, name: str | None = None) -> Articulation:
        """
        Build the articulation.

        Arguments:
            name: The name of the articulation.

        Returns:
            The built articulation.
        """
        for sim in self.__sims.values():
            if type(self.__sim_builders[sim.id]) is BaseArticulationBuilder:
                raise NotImplementedError(
                    f"{self.__sim_builders[sim.id].__class__.__name__} does not support "
                    "build."
                )
            articulation = self.__sim_builders[sim.id].build(name=name)
        return articulation
