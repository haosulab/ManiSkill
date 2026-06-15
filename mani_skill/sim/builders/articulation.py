from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from mani_skill.sim.builders.base_builder import BaseBuilder
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.pose import Pose

if TYPE_CHECKING:
    from mani_skill.sim.base_sim import BaseSim


class BaseArticulationBuilder(BaseBuilder, ABC):
    """Base articulation builder for building articulated objects in a simulation.
    Articulation builders for each simulator backend should inherit from this class."""

    initial_pose: Pose | None = None
    """The initial pose of the articulation's root link when it gets built and spawned into the
    simulation."""

    scene_idxs: list[int] | None = None
    """The list of scene indices to build this articulation in. If None, the articulation will
    be built in all scenes"""

    def set_scene_idxs(self, scene_idxs: list[int]):
        self.scene_idxs = scene_idxs
        return self

    @abstractmethod
    def build(self, name: str) -> Articulation:
        """
        Build the articulation.

        Arguments:
            name: The name of the articulation.

        Returns:
            The built articulation.
        """


class ArticulationBuilder(BaseArticulationBuilder):
    """Articulation builder for building articulated objects in a simulation.
    This is simulator independent and can be used to build articulations across different simulators
    simultaneously to support e.g. rendering in one simulator and running physics in another."""

    _sims: dict[str, BaseSim] = {}
    """dictionary of simulators that will be tracking this builder. There can be multiple simulators
    that track this builder in order to support using different simulators for physics and
    rendering."""

    _sim_builders: dict[str, BaseArticulationBuilder] = {}
    """dictionary mapping sim id to the corresponding articulation builder for that simulator."""

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
        self._sims[sim.id] = sim
        self._sim_builders[sim.id] = sim.create_articulation_builder()
        return self

    def _remove_sim(self, sim: BaseSim):
        """
        Remove a simulation backend that is tracking this builder.
        """
        self._sims.pop(sim.id)
        return self

    def build(self):
        pass
