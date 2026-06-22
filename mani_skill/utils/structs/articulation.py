from dataclasses import dataclass
from typing import Generic, TypeVar

from mani_skill.utils.structs.base import BaseStruct

T = TypeVar("T", bound=BaseStruct)


@dataclass
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
