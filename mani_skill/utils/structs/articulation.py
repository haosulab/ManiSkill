from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

from mani_skill.utils.structs.base import BaseStruct

T = TypeVar("T", bound=BaseStruct)


@dataclass(kw_only=True)
class Articulation(Generic[T]):
    merged: bool = False
    """
    Whether or not this articulation object is a merged articulation where it is managing many
    articulations with different DOFs.

    There are a number of caveats when it comes to merged articulations. While merging
    articulations means you can easily fetch padded qpos, qvel, etc. type data, a number of
    attributes and functions will make little sense and you should avoid using them unless you
    are an advanced user. In particular, the list of Links, Joints, their corresponding maps,
    net contact forces of multiple links, no longer make "sense"
    """

    name: str = ""
    """The name of the articulation. Must be unique within the scene."""

    @classmethod
    def merge(
        cls, articulations: list["Articulation"], name: str, merge_links: bool = False
    ):
        """
        Merge a list of articulations into a single articulation for easy access of data across
        multiple possibly different articulations.

        Args:
            articulations: A list of articulations objects to merge.
            name: The name of the merged articulation.
            merge_links: Whether to merge the links of the articulations. This is by default False
                as often times you merge articulations that have different number of links. Set
                this true if you want to try and merge articulations that have the same number of
                links.
        """
        articulation_cls: "Articulation" = articulations[0].__class__  # type: ignore
        return articulation_cls.merge(articulations, name, merge_links)

    def set_qpos(self, qpos: torch.Tensor):
        """
        Set the qpos of the articulation.
        """
        raise NotImplementedError()

    def set_qvel(self, qvel: torch.Tensor):
        """
        Set the qvel of the articulation.
        """
        raise NotImplementedError()

    def set_root_linear_velocity(self, velocity: torch.Tensor):
        """
        Set the root linear velocity of the articulation.
        """
        raise NotImplementedError()

    def set_root_angular_velocity(self, velocity: torch.Tensor):
        """
        Set the root angular velocity of the articulation.
        """
        raise NotImplementedError()

    @property
    def max_dof(self) -> int:
        """
        The maximum number of DOFs of the articulation.
        """
        raise NotImplementedError()
