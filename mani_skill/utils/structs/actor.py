from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from mani_skill.utils.structs.base import BaseStruct

T = TypeVar("T", bound=BaseStruct)


@dataclass
class Actor(Generic[T]):
    """
    The actor class manages rigid body objects in simulation.
    """

    hidden: bool = False
    """Whether this actor is hidden from any camera sensors."""

    merged: bool = False
    """Whether this object is a view of other actors as a result of Actor.merge."""

    name: str = ""
    """The name of the actor. Must be unique within the scene."""
