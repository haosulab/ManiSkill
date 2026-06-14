from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mani_skill.utils.structs.base import BaseStruct

if TYPE_CHECKING:
    pass


@dataclass
class Actor(BaseStruct):
    """
    The actor class manages a rigid body object in simulation.
    """

    hidden: bool = False
    """Whether this actor is hidden from any camera sensors."""

    merged: bool = False
    """Whether this object is a view of other actors as a result of Actor.merge."""

    name: str | None = None
    """The name of the actor."""
