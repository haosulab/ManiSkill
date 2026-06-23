from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, TypedDict

from mani_skill.sim.builders.actor import BaseActorBuilder
from mani_skill.sim.builders.articulation import BaseArticulationBuilder
from mani_skill.utils.structs.articulation import Articulation

if TYPE_CHECKING:
    from mani_skill.sim.base_sim import BaseSim


class ParsedURDFData(TypedDict):
    articulation_builders: list[BaseArticulationBuilder]
    actor_builders: list[BaseActorBuilder]
    cameras: list[Any]


class BaseURDFLoader:
    """Base class for URDF loaders"""

    scene_idxs: list[int] | None = None
    """The list of scene indices to build this actor in. If None, the actor will be
    built in all scenes."""

    __sims: dict[str, BaseSim] = {}
    """dictionary of simulators that will be tracking this builder. There can be multiple
    simulators that track this builder in order to support using different simulators for
    physics and rendering."""

    __sim_builders: dict[str, "BaseURDFLoader"] = {}
    """dictionary mapping sim id to the corresponding articulation builder for that simulator.
    """

    sim: BaseSim
    """The simulation backend this loader loads in"""

    def _add_sim(self, sim: BaseSim):
        """
        Add a simulation backend that should track this loader. Whenever this URDF is
        loaded, the simulator backend will include this URDF in its state and compile
        it in the scene.

        Args:
            sim: The simulation backend to add.
        """
        self.__sims[sim.id] = sim
        self.__sim_builders[sim.id] = sim.create_urdf_loader()
        self.__sim_builders[sim.id].sim = sim
        return self

    @property
    def fix_root_link(self) -> bool:
        """Whether to fix the root link of the URDF."""
        return self.__sim_builders[next(iter(self.__sims.values())).id].fix_root_link

    @fix_root_link.setter
    def fix_root_link(self, fix_root_link: bool):
        """Whether to fix the root link of the URDF."""
        for sim in self.__sims.values():
            if type(self.__sim_builders[sim.id]) is BaseURDFLoader:
                raise NotImplementedError(
                    f"{self.__sim_builders[sim.id].__class__.__name__} does not have "
                    "a fix_root_link attribute."
                )
            self.__sim_builders[sim.id].fix_root_link = fix_root_link

    @property
    def load_multiple_collisions_from_file(self) -> bool:
        """Whether to load multiple collisions from the file."""
        return self.__sim_builders[
            next(iter(self.__sims.values())).id
        ].load_multiple_collisions_from_file

    @load_multiple_collisions_from_file.setter
    def load_multiple_collisions_from_file(
        self, load_multiple_collisions_from_file: bool
    ):
        """Whether to load multiple collisions from the file."""
        for sim in self.__sims.values():
            if type(self.__sim_builders[sim.id]) is BaseURDFLoader:
                raise NotImplementedError(
                    f"{self.__sim_builders[sim.id].__class__.__name__} does not have "
                    "a load_multiple_collisions_from_file attribute."
                )
            self.__sim_builders[
                sim.id
            ].load_multiple_collisions_from_file = load_multiple_collisions_from_file

    @property
    def disable_self_collisions(self) -> bool:
        """Whether to disable self collisions of the URDF."""
        return self.__sim_builders[
            next(iter(self.__sims.values())).id
        ].disable_self_collisions

    @disable_self_collisions.setter
    def disable_self_collisions(self, disable_self_collisions: bool):
        """Whether to disable self collisions of the URDF."""
        for sim in self.__sims.values():
            if type(self.__sim_builders[sim.id]) is BaseURDFLoader:
                raise NotImplementedError(
                    f"{self.__sim_builders[sim.id].__class__.__name__} does not have "
                    "a disable_self_collisions attribute."
                )
                self.__sim_builders[
                    sim.id
                ].disable_self_collisions = disable_self_collisions

    @property
    def name(self) -> str:
        """The name of the URDF."""
        return self.__sim_builders[next(iter(self.__sims.values())).id].name

    @name.setter
    def name(self, name: str):
        """The name of the URDF."""
        for sim in self.__sims.values():
            if type(self.__sim_builders[sim.id]) is BaseURDFLoader:
                raise NotImplementedError(
                    f"{self.__sim_builders[sim.id].__class__.__name__} does not have "
                    "a name attribute."
                )
            self.__sim_builders[sim.id].name = name

    @property
    def scale(self) -> float:
        """The scale of the URDF."""
        return self.__sim_builders[next(iter(self.__sims.values())).id].scale

    @scale.setter
    def scale(self, scale: float):
        """The scale of the URDF."""
        for sim in self.__sims.values():
            if type(self.__sim_builders[sim.id]) is BaseURDFLoader:
                raise NotImplementedError(
                    f"{self.__sim_builders[sim.id].__class__.__name__} does not have "
                    "a scale attribute."
                )
            self.__sim_builders[sim.id].scale = scale

    def parse(
        self,
        urdf_file: str,
        srdf_file: str | None = None,
        package_dir: str | None = None,
    ) -> ParsedURDFData:
        """
        Parses a given URDF and optionally SRDF file and returns a dictionary of all
        found articulation and actor builders

        Args:
            urdf_file: The path to the URDF file to parse.
            srdf_file: The path to the SRDF file to parse. If None, no SRDF will be
                parsed.
            package_dir: The directory to resolve package paths in the URDF file. If
                None, no package paths will be resolved.

        Returns:
            A dictionary of all found articulation and actor builders
        """
        for sim in self.__sims.values():
            parsed = self.__sim_builders[sim.id].parse(
                urdf_file, srdf_file, package_dir
            )
            # TODO (stao): we may just assume there can only ever be one physics sim
            # and one render sim
        return parsed

    @abstractmethod
    def load(
        self,
        urdf_file: str,
        srdf_file: str | None = None,
        package_dir: str | None = None,
    ) -> Articulation:
        """
        Loads a given URDF and optionally SRDF file and returns the first articulation
        found and builds it.

        Args:
            urdf_file: The path to the URDF file to load.
            srdf_file: The path to the SRDF file to load. If None, no SRDF will be
                loaded.
            package_dir: The directory to resolve package paths in the URDF file. If
                None, no package paths will be resolved.

        Returns:
            A single articulation loaded from the URDF file
        """
        pass

    def set_link_material(
        self,
        link_name: str,
        static_friction: float,
        dynamic_friction: float,
        restitution: float,
    ):
        """
        Sets the material for a link.

        Args:
            link_name: The name of the link to set the material for.
            static_friction: The static friction coefficient.
            dynamic_friction: The dynamic friction coefficient.
            restitution: The restitution coefficient.
        """
        for sim in self.__sims.values():
            self.__sim_builders[sim.id].set_link_material(
                link_name, static_friction, dynamic_friction, restitution
            )

    def set_material(
        self, static_friction: float, dynamic_friction: float, restitution: float
    ):
        """
        Sets the material for the URDF.
        """
        for sim in self.__sims.values():
            self.__sim_builders[sim.id].set_material(
                static_friction, dynamic_friction, restitution
            )

    def set_density(self, density: float):
        """
        Sets the density for the URDF.
        """
        for sim in self.__sims.values():
            self.__sim_builders[sim.id].set_density(density)

    # TODO (stao): patch radius might not be supported by all simulator backends

    def set_link_patch_radius(self, link_name: str, patch_radius: float):
        """
        Sets the patch radius for a link.

        Args:
            link_name: The name of the link to set the patch radius for.
            patch_radius: The patch radius.
        """
        for sim in self.__sims.values():
            self.__sim_builders[sim.id].set_link_patch_radius(link_name, patch_radius)

    def set_link_min_patch_radius(self, link_name: str, min_patch_radius: float):
        """
        Sets the minimum patch radius for a link.

        Args:
            link_name: The name of the link to set the minimum patch radius for.
            min_patch_radius: The minimum patch radius.
        """
        for sim in self.__sims.values():
            self.__sim_builders[sim.id].set_link_min_patch_radius(
                link_name, min_patch_radius
            )
