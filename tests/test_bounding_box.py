import numpy as np
import pytest
import sapien
import sapien.physx as physx
import trimesh
from unittest import mock

from mani_skill.utils.geometry import BoundingBox
from mani_skill.utils.geometry.geometry import (
    get_axis_aligned_bbox_for_actor,
    get_axis_aligned_bbox_for_articulation,
    get_local_axis_aligned_bbox_for_link,
)


def make_box_actor(extents, pose):
    """Create a sapien entity with a box collision mesh without needing a render system."""
    verts = np.asarray(
        trimesh.creation.box(extents=extents).vertices, dtype=np.float32
    )
    comp = physx.PhysxRigidDynamicComponent()
    comp.attach(
        physx.PhysxCollisionShapeConvexMesh(
            vertices=verts,
            scale=np.ones(3, dtype=np.float32),
            material=physx.PhysxMaterial(0.5, 0.5, 0.0),
        )
    )
    entity = sapien.Entity()
    entity.add_component(comp)
    entity.pose = pose
    return entity


def test_from_points():
    pts = np.array([[0, 0, 0], [1, 2, 3], [0.5, -1, 2]], dtype=np.float32)
    bbox = BoundingBox.from_points(pts)
    assert np.allclose(bbox.mins, [0, -1, 0])
    assert np.allclose(bbox.maxs, [1, 2, 3])


def test_center_sizes_corners():
    bbox = BoundingBox(mins=[-1, -2, -3], maxs=[1, 2, 3])
    assert np.allclose(bbox.center, [0, 0, 0])
    assert np.allclose(bbox.sizes, [2, 4, 6])
    assert bbox.corners.shape == (8, 3)
    assert np.allclose(bbox.corners.min(0), bbox.mins)
    assert np.allclose(bbox.corners.max(0), bbox.maxs)


def test_merge():
    a = BoundingBox(mins=[0, 0, 0], maxs=[1, 1, 1])
    b = BoundingBox(mins=[2, -1, 0.5], maxs=[3, 0.5, 4])
    m = a.merged(b)
    assert np.allclose(m.mins, [0, -1, 0])
    assert np.allclose(m.maxs, [3, 1, 4])
    # merged box contains both original boxes
    for bbox in (a, b):
        assert np.allclose(m.mins, np.minimum(m.mins, bbox.mins))
        assert np.allclose(m.maxs, np.maximum(m.maxs, bbox.maxs))
    # multi-box merge gives the same result
    m2 = BoundingBox.from_boxes([a, b])
    assert m == m2


def test_transformed_translation_and_rotation():
    bbox = BoundingBox(mins=[-0.1, -0.1, -0.1], maxs=[0.1, 0.1, 0.1])
    # pure translation
    moved = bbox.transformed(sapien.Pose(p=[1, 2, 3]))
    assert np.allclose(moved.mins, [0.9, 1.9, 2.9], atol=1e-6)
    assert np.allclose(moved.maxs, [1.1, 2.1, 3.1], atol=1e-6)
    # pure 90 degree rotation around z: x-extent and y-extent swap
    q = sapien.Pose(q=[np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    box = BoundingBox(mins=[-1, -0.5, 0], maxs=[1, 0.5, 2])
    rot = box.transformed(q)
    assert np.allclose(rot.sizes[:2], [1.0, 2.0], atol=1e-5)
    # the rotated box must still enclose the transformed corners
    corners = box.corners @ q.to_transformation_matrix()[:3, :3].T
    assert np.all(rot.mins <= corners.min(0) + 1e-5)
    assert np.all(rot.maxs >= corners.max(0) - 1e-5)


def test_contains_and_intersects():
    a = BoundingBox(mins=[0, 0, 0], maxs=[1, 1, 1])
    assert a.contains([0.5, 0.5, 0.5])
    assert a.contains([0, 0, 0])
    assert not a.contains([1.5, 0.5, 0.5])
    b = BoundingBox(mins=[0.5, 0.5, 0.5], maxs=[2, 2, 2])
    c = BoundingBox(mins=[3, 3, 3], maxs=[4, 4, 4])
    assert a.intersects(b)
    assert b.intersects(a)
    assert not a.intersects(c)


def test_from_actor_matches_functional_api():
    extents = [0.2, 0.4, 0.6]
    actor = make_box_actor(extents, sapien.Pose(p=[1, 2, 3]))
    mins, maxs = get_axis_aligned_bbox_for_actor(actor)
    bbox = BoundingBox.from_actor(actor)
    assert np.allclose(bbox.mins, mins)
    assert np.allclose(bbox.maxs, maxs)
    # a box of extents centered at (1,2,3)
    assert np.allclose(bbox.center, [1, 2, 3], atol=1e-5)
    assert np.allclose(bbox.sizes, extents, atol=1e-5)
    # actor struct wrapper delegates to the same values
    assert BoundingBox.create(actor) == bbox


def test_create_unwraps_structs():
    """BoundingBox.create works for raw sapien objects and simple structs alike."""
    actor = make_box_actor([0.2, 0.4, 0.6], sapien.Pose(p=[0, 0, 0]))
    assert BoundingBox.create(actor) == BoundingBox.from_actor(actor)
    with pytest.raises(TypeError):
        BoundingBox.create("not a bounding box source")


class FakeShape:
    """Duck-typed stand-in for PhysxCollisionShapeConvexMesh."""

    def __init__(self, vertices):
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.scale = np.ones(3, dtype=np.float32)

    def get_local_pose(self):
        return sapien.Pose()


class FakeLink:
    def __init__(self, vertices, pose):
        self._shapes = [FakeShape(vertices)]
        self.pose = pose

    def get_collision_shapes(self):
        return self._shapes


class FakeArticulation:
    def __init__(self, links):
        self._links = links

    def get_links(self):
        return self._links


def test_from_articulation_merges_link_boxes():
    """Two links offset in the world produce a merged articulation-level box."""
    verts = np.asarray(
        trimesh.creation.box(extents=[0.2, 0.2, 0.2]).vertices, dtype=np.float32
    )
    links = [
        FakeLink(verts, sapien.Pose(p=[0, 0, 0])),
        FakeLink(verts, sapien.Pose(p=[1, 0, 0])),
    ]
    art = FakeArticulation(links)

    # the mesh-type assert is bypassed with duck-typed shapes
    with mock.patch.object(
        physx, "PhysxCollisionShapeConvexMesh", FakeShape
    ):
        bbox = BoundingBox.from_articulation(art)
        mins, maxs = get_axis_aligned_bbox_for_articulation(art)
    assert np.allclose(bbox.mins, [-0.1, -0.1, -0.1], atol=1e-5)
    assert np.allclose(bbox.maxs, [1.1, 0.1, 0.1], atol=1e-5)
    assert np.allclose(bbox.mins, mins)
    assert np.allclose(bbox.maxs, maxs)


def test_local_link_bbox_matches_functional_api():
    verts = np.asarray(
        trimesh.creation.box(extents=[0.2, 0.4, 0.6]).vertices, dtype=np.float32
    )
    with mock.patch.object(
        physx, "PhysxCollisionShapeConvexMesh", FakeShape
    ):
        link = FakeLink(verts, sapien.Pose(p=[5, 5, 5]))
        mins, maxs = get_local_axis_aligned_bbox_for_link(link)
        bbox = BoundingBox.from_link(link)
    assert np.allclose(bbox.mins, mins)
    assert np.allclose(bbox.maxs, maxs)
    assert np.allclose(bbox.sizes, [0.2, 0.4, 0.6], atol=1e-5)
