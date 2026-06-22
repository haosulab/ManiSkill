from abc import abstractmethod
from typing import Any, TypedDict

from mani_skill.sim.builders.actor import BaseActorBuilder
from mani_skill.sim.builders.articulation import BaseArticulationBuilder
from mani_skill.utils.structs.articulation import Articulation


class ParsedURDFData(TypedDict):
    articulation_builders: list[BaseArticulationBuilder]
    actor_builders: list[BaseActorBuilder]
    cameras: list[Any]


class BaseURDFLoader:
    """Base class for URDF loaders"""

    @abstractmethod
    def parse(
        self,
        urdf_file: str,
        srdf_file: str | None = None,
        package_dir: str | None = None,
    ) -> ParsedURDFData:
        """
        Parses a given URDF and optionally SRDF file and returns a dictionary of all found
        articulation and actor builders

        Args:
            urdf_file: The path to the URDF file to parse.
            srdf_file: The path to the SRDF file to parse. If None, no SRDF will be parsed.
            package_dir: The directory to resolve package paths in the URDF file. If None, no
                package paths will be resolved.

        Returns:
            A dictionary of all found articulation and actor builders
        """
        pass

    @abstractmethod
    def load(
        self,
        urdf_file: str,
        srdf_file: str | None = None,
        package_dir: str | None = None,
    ) -> Articulation:
        """
        Loads a given URDF and optionally SRDF file and returns the first articulation found and
        builds it.

        Args:
            urdf_file: The path to the URDF file to load.
            srdf_file: The path to the SRDF file to load. If None, no SRDF will be loaded.
            package_dir: The directory to resolve package paths in the URDF file. If None, no
                package paths will be resolved.

        Returns:
            A single articulation loaded from the URDF file
        """
        pass
