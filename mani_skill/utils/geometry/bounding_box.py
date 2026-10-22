"""Object oriented bounding box utilities for actors and articulations.

This module provides :class:`BoundingBox`, a small object oriented wrapper
around axis-aligned bounding boxes (AABBs) that supports merging, transforms,
and simple collision/containment queries. The old functional utilities in
:mod:`mani_skill.utils.geometry.geometry`
(``get_axis_aligned_bbox_for_actor``, ``get_axis_aligned_bbox_for_articulation``
and ``get_local_axis_aligned_bbox_for_link``) are kept as thin delegates to
this class for backwards compatibility.
"""

from dataclasses import dataclass
from typing import List, Union

import numpy as np
import numpy.typing as npt
import sapien
import sapien.physx as physx

ArrayLike = npt.ArrayLike

# Objects (sapien entities, physx links/articulations, or ManiSkill structs
# such as mani_skill.utils.structs.actor.Actor) that a BoundingBox can be
# constructed from.
BoundingBoxSource = Union[
    sapien.Entity,
    physx.PhysxArticulation,
    physx.PhysxArticulationLinkComponent,
]


def _unwrap_struct(obj):
    """Unwrap ManiSkill structs (objects with a ``_objs`` multi-view list).

    For structs that manage multiple objects across sub-scenes we use the
    first object and raise if there is more than one, since a single AABB
    would be ill-defined for objects living in different scenes.
    """
    objs = getattr(obj, "_objs", None)
    if objs is None:
        return obj
    if len(objs) != 1:
        raise ValueError(
            "Cannot compute a single bounding box for a struct managing "
            f"{len(objs)} objects across sub-scenes. Use the underlying "
            "per-scene objects instead."
        )
    return objs[0]


@dataclass
class BoundingBox:
    """An axis-aligned bounding box defined by its minimum and maximum corners."""

    mins: np.ndarray
    """Minimum corner of the bounding box, shape (3,)."""
    maxs: np.ndarray
    """Maximum corner of the bounding box, shape (3,)."""

    def __post_init__(self):
        self.mins = np.asarray(self.mins, dtype=np.float32)
        self.maxs = np.asarray(self.maxs, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_points(cls, points: np.ndarray) -> "BoundingBox":
        """Create a bounding box tightly enclosing the given points.

        Args:
            points: array of shape (N, 3).
        """
        points = np.asarray(points, dtype=np.float32)
        return cls(mins=points.min(0), maxs=points.max(0))

    @classmethod
    def from_boxes(cls, boxes: List["BoundingBox"]) -> "BoundingBox":
        """Create a bounding box enclosing all the given bounding boxes."""
        return cls(
            mins=np.min([b.mins for b in boxes], axis=0),
            maxs=np.max([b.maxs for b in boxes], axis=0),
        )

    @classmethod
    def from_link(
        cls, link: physx.PhysxArticulationLinkComponent
    ) -> "BoundingBox":
        """Create a bounding box for a link's collision shapes in the link's local frame."""
        return cls.from_points(_collect_link_vertices(link))

    @classmethod
    def from_actor(cls, actor: sapien.Entity) -> "BoundingBox":
        """Create a bounding box for an actor's collision shapes in the world frame."""
        return cls.from_points(_collect_actor_vertices(actor))

    @classmethod
    def from_articulation(cls, art: physx.PhysxArticulation) -> "BoundingBox":
        """Create a bounding box for an articulation's collision shapes in the world frame."""
        boxes = []
        for link in art.get_links():
            link_box = cls.from_link(link)
            # move the link-local box into the world frame
            boxes.append(link_box.transformed(link.pose))
        return cls.from_boxes(boxes)

    @classmethod
    def create(cls, source: BoundingBoxSource) -> "BoundingBox":
        """Create a bounding box from an actor, articulation, link, or ManiSkill struct."""
        source = _unwrap_struct(source)
        if isinstance(source, physx.PhysxArticulation):
            return cls.from_articulation(source)
        if isinstance(source, physx.PhysxArticulationLinkComponent):
            return cls.from_link(source)
        if isinstance(source, sapien.Entity):
            return cls.from_actor(source)
        raise TypeError(f"Unsupported bounding box source: {type(source)}")

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def center(self) -> np.ndarray:
        """Center of the bounding box, shape (3,)."""
        return (self.mins + self.maxs) / 2

    @property
    def sizes(self) -> np.ndarray:
        """Sizes (extent) of the bounding box along each axis, shape (3,)."""
        return self.maxs - self.mins

    @property
    def corners(self) -> np.ndarray:
        """The 8 corners of the bounding box, shape (8, 3)."""
        c = np.array(
            [
                [x, y, z]
                for x in (self.mins[0], self.maxs[0])
                for y in (self.mins[1], self.maxs[1])
                for z in (self.mins[2], self.maxs[2])
            ],
            dtype=np.float32,
        )
        return c

    # ------------------------------------------------------------------ #
    # Operations
    # ------------------------------------------------------------------ #
    def merged(self, other: "BoundingBox") -> "BoundingBox":
        """Return a new bounding box enclosing both this box and ``other``."""
        return type(self).from_boxes([self, other])

    def transformed(self, pose: sapien.Pose) -> "BoundingBox":
        """Return the axis-aligned bounding box of this box under a rigid pose.

        The box is first converted to its 8 corners, the corners are
        transformed by ``pose``, and a new tight AABB around the transformed
        corners is returned. Note the result is an axis-aligned bound of the
        rotated box, not the rotated oriented box itself.
        """
        T = pose.to_transformation_matrix()
        transformed_corners = self.corners @ T[:3, :3].T + T[:3, 3]
        return type(self).from_points(transformed_corners)

    def contains(self, point: np.ndarray) -> bool:
        """Whether a point (shape (3,)) lies inside the bounding box (inclusive)."""
        point = np.asarray(point)
        return bool(np.all(point >= self.mins) and np.all(point <= self.maxs))

    def intersects(self, other: "BoundingBox") -> bool:
        """Whether this bounding box overlaps another bounding box."""
        return bool(
            np.all(self.mins <= other.maxs) and np.all(other.mins <= self.maxs)
        )

    def __eq__(self, other):
        if not isinstance(other, BoundingBox):
            return NotImplemented
        return (
            np.allclose(self.mins, other.mins) and np.allclose(self.maxs, other.maxs)
        )


def _collect_link_vertices(link: physx.PhysxArticulationLinkComponent) -> np.ndarray:
    """Collect collision shape vertices of a link in the link's local frame."""
    all_vertices = []
    for s in link.get_collision_shapes():
        assert isinstance(s, physx.PhysxCollisionShapeConvexMesh)
        p = s.get_local_pose()
        T = p.to_transformation_matrix()
        vertices = s.vertices * s.scale
        vertices = vertices @ T[:3, :3].T + T[:3, 3]
        all_vertices.append(vertices)
    return np.vstack(all_vertices)


def _collect_actor_vertices(actor: sapien.Entity) -> np.ndarray:
    """Collect collision shape vertices of an actor in the world frame."""
    all_vertices = []
    for shape in actor.find_component_by_type(
        physx.PhysxRigidDynamicComponent
    ).get_collision_shapes():
        assert isinstance(shape, physx.PhysxCollisionShapeConvexMesh)
        scaled_vertices = shape.vertices * shape.scale
        local_pose = shape.get_local_pose()
        mat = (actor.get_pose() * local_pose).to_transformation_matrix()
        world_vertices = scaled_vertices @ (mat[:3, :3].T) + mat[:3, 3]
        all_vertices.append(world_vertices)
    return np.vstack(all_vertices)
