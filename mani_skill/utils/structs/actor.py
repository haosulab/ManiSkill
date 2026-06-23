from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

import torch

from mani_skill.utils.structs.base import BaseStruct
from mani_skill.utils.structs.pose import Pose

T = TypeVar("T", bound=BaseStruct)

if TYPE_CHECKING:
    from mani_skill.sim.base_sim import BaseSim


@dataclass(kw_only=True)
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

    body_type: Literal["kinematic", "static", "dynamic"] = "dynamic"
    """The type of the body of the actor."""

    @abstractmethod
    def create_from_entities(
        cls,
        entities: list[T],
        scene_idxs: torch.Tensor,
        sim: BaseSim,
        shared_name: str | None = None,
    ):
        """
        Create an actor from a list of entities.
        """
        raise NotImplementedError()

    @classmethod
    def merge(cls, actors: list["Actor"], name: str):
        """
        Merge actors together under one view so that they can all be managed by one python
        dataclass object. This can be useful for e.g. randomizing the asset loaded into a
        task and being able to do object.pose to fetch the pose of all randomized assets or
        object.set_pose to change the pose of each of the different assets, despite the
        assets not being uniform across all sub-scenes.

        For example usage of this method, see mani_skill/envs/tasks/pick_single_ycb.py

        Args:
            actors (list[Actor]): The actors to merge into one actor object to manage
            name (str): A new name to give the merged actors. If none, the name will default
                to the first actor's name
        """
        actor_cls: "Actor" = actors[0].__class__  # type: ignore
        return actor_cls.merge(actors, name)

    def hide_visual(self):
        """
        Hide the visuals of the actor.
        """
        raise NotImplementedError()

    def show_visual(self):
        """
        Show the visuals of the actor.
        """
        raise NotImplementedError()

    @property
    def linear_velocity(self) -> torch.Tensor:
        """
        Get the linear velocity of the actor.
        """
        raise NotImplementedError()

    @linear_velocity.setter
    def linear_velocity(self, velocity: torch.Tensor):
        """
        Set the linear velocity of the actor.
        """
        raise NotImplementedError()

    def set_linear_velocity(self, velocity: torch.Tensor):
        """
        Set the linear velocity of the actor.
        """
        self.linear_velocity = velocity

    @property
    def angular_velocity(self) -> torch.Tensor:
        """
        Get the angular velocity of the actor.
        """
        raise NotImplementedError()

    @angular_velocity.setter
    def angular_velocity(self, velocity: torch.Tensor):
        """
        Set the angular velocity of the actor.
        """
        raise NotImplementedError()

    def set_angular_velocity(self, velocity: torch.Tensor):
        """
        Set the angular velocity of the actor.
        """
        self.angular_velocity = velocity

    @property
    def pose(self) -> Pose:
        """
        Get the pose of the actor.
        """
        raise NotImplementedError()

    @pose.setter
    def pose(self, pose: Pose):
        """
        Set the pose of the actor.
        """

    def set_pose(self, pose: Pose):
        """
        Set the pose of the actor.
        """
        self.pose = pose
