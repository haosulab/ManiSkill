from typing import Dict

import numpy as np
import sapien.core as sapien
from sapien.core import Actor, Articulation, Link, Pose
from scipy.spatial.transform import Rotation

from mani_skill2.utils.bounding_cylinder import aabc


def sample_on_unit_sphere(rng):
    """
    Algo from http://corysimon.github.io/articles/uniformdistn-on-sphere/
    """
    v = np.zeros(3)
    while np.linalg.norm(v) < 1e-4:
        v[0] = rng.normal()  # random standard normal
        v[1] = rng.normal()
        v[2] = rng.normal()

    v = v / np.linalg.norm(v)
    return v


def sample_on_unit_circle(rng):
    v = np.zeros(2)
    while np.linalg.norm(v) < 1e-4:
        v[0] = rng.normal()  # random standard normal
        v[1] = rng.normal()

    v = v / np.linalg.norm(v)
    return v


def rotation_between_vec(a, b):  # from a to b
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    axis = np.cross(a, b)
    axis = axis / np.linalg.norm(axis)  # norm might be 0
    angle = np.arccos(a @ b)
    R = Rotation.from_rotvec(axis * angle)
    return R


def angle_between_vec(a, b):  # from a to b
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    angle = np.arccos(a @ b)
    return angle


def wxyz_to_xyzw(q):
    return np.concatenate([q[1:4], q[0:1]])


def xyzw_to_wxyz(q):
    return np.concatenate([q[3:4], q[0:3]])


def rotate_2d_vec_by_angle(vec, theta):
    rot_mat = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return rot_mat @ vec


def angle_distance(q0: sapien.Pose, q1: sapien.Pose):
    qd = (q0.inv() * q1).q
    return 2 * np.arctan2(np.linalg.norm(qd[1:]), qd[0]) / np.pi


def get_axis_aligned_bbox_for_articulation(art: Articulation):
    mins = np.array([np.inf, np.inf, np.inf])
    maxs = -mins
    for link in art.get_links():
        lp = link.pose
        for s in link.get_collision_shapes():
            p = lp * s.get_local_pose()
            T = p.to_transformation_matrix()
            vertices = s.geometry.vertices * s.geometry.scale
            vertices = vertices @ T[:3, :3].T + T[:3, 3]
            mins = np.minimum(mins, vertices.min(0))
            maxs = np.maximum(maxs, vertices.max(0))
    return mins, maxs


def get_axis_aligned_bbox_for_actor(actor: Actor):
    mins = np.ones(3) * np.inf
    maxs = -mins

    for shape in actor.get_collision_shapes():  # this is CollisionShape
        scaled_vertices = shape.geometry.vertices * shape.geometry.scale
        local_pose = shape.get_local_pose()
        mat = (actor.get_pose() * local_pose).to_transformation_matrix()
        world_vertices = scaled_vertices @ (mat[:3, :3].T) + mat[:3, 3]
        mins = np.minimum(mins, world_vertices.min(0))
        maxs = np.maximum(maxs, world_vertices.max(0))

    return mins, maxs


def get_local_axis_aligned_bbox_for_link(link: Link):
    mins = np.array([np.inf, np.inf, np.inf])
    maxs = -mins
    for s in link.get_collision_shapes():
        p = s.get_local_pose()
        T = p.to_transformation_matrix()
        vertices = s.geometry.vertices * s.geometry.scale
        vertices = vertices @ T[:3, :3].T + T[:3, 3]
        mins = np.minimum(mins, vertices.min(0))
        maxs = np.maximum(maxs, vertices.max(0))
    return mins, maxs


def get_local_aabc_for_actor(actor):
    all_vertices = []
    for s in actor.get_collision_shapes():
        p = s.get_local_pose()
        T = p.to_transformation_matrix()
        vertices = s.geometry.vertices * s.geometry.scale
        vertices = vertices @ T[:3, :3].T + T[:3, 3]
        all_vertices.append(vertices)
    vertices = np.vstack(all_vertices)
    return aabc(vertices)


def transform_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    assert H.shape == (4, 4), H.shape
    assert pts.ndim == 2 and pts.shape[1] == 3, pts.shape
    return pts @ H[:3, :3].T + H[:3, 3]


def invert_transform(H: np.ndarray):
    assert H.shape[-2:] == (4, 4), H.shape
    H_inv = H.copy()
    R_T = np.swapaxes(H[..., :3, :3], -1, -2)
    H_inv[..., :3, :3] = R_T
    H_inv[..., :3, 3:] = -R_T @ H[..., :3, 3:]
    return H_inv


def get_oriented_bounding_box_for_2d_points(
    points_2d: np.ndarray, resolution=0.0
) -> Dict:
    assert len(points_2d.shape) == 2 and points_2d.shape[1] == 2
    if resolution > 0.0:
        points_2d = np.round(points_2d / resolution) * resolution
        points_2d = np.unique(points_2d, axis=0)
    ca = np.cov(points_2d, y=None, rowvar=0, bias=1)

    v, vect = np.linalg.eig(ca)
    tvect = np.transpose(vect)

    # use the inverse of the eigenvectors as a rotation matrix and
    # rotate the points so they align with the x and y axes
    ar = np.dot(points_2d, np.linalg.inv(tvect))

    # get the minimum and maximum x and y
    mina = np.min(ar, axis=0)
    maxa = np.max(ar, axis=0)
    half_size = (maxa - mina) * 0.5

    # the center is just half way between the min and max xy
    center = mina + half_size
    # get the 4 corners by subtracting and adding half the bounding boxes height and width to the center
    corners = np.array(
        [
            center + [-half_size[0], -half_size[1]],
            center + [half_size[0], -half_size[1]],
            center + [half_size[0], half_size[1]],
            center + [-half_size[0], half_size[1]],
        ]
    )

    # use the the eigenvectors as a rotation matrix and
    # rotate the corners and the centerback
    corners = np.dot(corners, tvect)
    center = np.dot(center, tvect)

    return {"center": center, "half_size": half_size, "axes": vect, "corners": corners}
