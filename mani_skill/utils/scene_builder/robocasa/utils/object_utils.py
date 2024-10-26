import numpy as np
from transforms3d.euler import euler2mat


def get_rel_transform(fixture_A, fixture_B):
    """
    Gets fixture_B's position and rotation relative to fixture_A's frame
    """
    A_trans = np.array(fixture_A.pos)
    B_trans = np.array(fixture_B.pos)

    # A_rot = np.array([0, 0, fixture_A.rot])
    # B_rot = np.array([0, 0, fixture_B.rot])

    A_mat = euler2mat(0, 0, fixture_A.rot)
    B_mat = euler2mat(0, 0, fixture_B.rot)

    T_WA = np.vstack((np.hstack((A_mat, A_trans[:, None])), [0, 0, 0, 1]))
    T_WB = np.vstack((np.hstack((B_mat, B_trans[:, None])), [0, 0, 0, 1]))

    T_AB = np.matmul(np.linalg.inv(T_WA), T_WB)

    return T_AB[:3, 3], T_AB[:3, :3]


def get_fixture_to_point_rel_offset(fixture, point):
    """
    get offset relative to fixture's frame, given a global point
    """
    global_offset = point - fixture.pos
    T_WF = euler2mat(0, 0, fixture.rot)
    rel_offset = np.matmul(np.linalg.inv(T_WF), global_offset)
    return rel_offset


def get_pos_after_rel_offset(fixture, offset):
    """
    get global position of a fixture, after applying offset relative to center of fixture
    """
    fixture_mat = euler2mat(0, 0, fixture.rot)

    return fixture.pos + np.dot(fixture_mat, offset)


def obj_in_region(
    obj,
    obj_pos,
    obj_quat,
    p0,
    px,
    py,
    pz=None,
):
    """
    check if object is in the region defined by the points.
    Uses either the objects bounding box or the object's horizontal radius
    """
    from mani_skill.utils.scene_builder.robocasa.fixtures.fixture import Fixture

    if isinstance(obj, Fixture):
        obj_points = obj.get_bbox_points(trans=obj_pos, rot=obj_quat)
    else:
        radius = obj.horizontal_radius
        obj_points = obj_pos + np.array(
            [
                [radius, 0, 0],
                [-radius, 0, 0],
                [0, radius, 0],
                [0, -radius, 0],
            ]
        )

    u = px - p0
    v = py - p0
    w = pz - p0 if pz is not None else None

    for point in obj_points:
        check1 = np.dot(u, p0) <= np.dot(u, point) <= np.dot(u, px)
        check2 = np.dot(v, p0) <= np.dot(v, point) <= np.dot(v, py)

        if not check1 or not check2:
            return False

        if w is not None:
            check3 = np.dot(w, p0) <= np.dot(w, point) <= np.dot(w, pz)
            if not check3:
                return False

    return True


def point_in_fixture(point, fixture, only_2d=False):
    """
    check if point is inside of the exterior bounding boxes of the fixture

    Args:
        point (np.array): point to check

        fixture (Fixture): fixture object

        only_2d (bool): whether to check only in 2D
    """
    p0, px, py, pz = fixture.get_ext_sites(relative=False)
    th = 0.00
    u = px - p0
    v = py - p0
    w = pz - p0
    check1 = np.dot(u, p0) - th <= np.dot(u, point) <= np.dot(u, px) + th
    check2 = np.dot(v, p0) - th <= np.dot(v, point) <= np.dot(v, py) + th
    check3 = np.dot(w, p0) - th <= np.dot(w, point) <= np.dot(w, pz) + th

    if only_2d:
        return check1 and check2
    else:
        return check1 and check2 and check3


def objs_intersect(
    obj,
    obj_pos,
    obj_quat,
    other_obj,
    other_obj_pos,
    other_obj_quat,
):
    """
    check if two objects intersect
    """
    from mani_skill.utils.scene_builder.robocasa.fixtures.fixture import Fixture
    from mani_skill.utils.scene_builder.robocasa.objects.objects import MJCFObject

    bbox_check = (isinstance(obj, MJCFObject) or isinstance(obj, Fixture)) and (
        isinstance(other_obj, MJCFObject) or isinstance(other_obj, Fixture)
    )
    if bbox_check:
        obj_points = obj.get_bbox_points(trans=obj_pos, rot=obj_quat)
        other_obj_points = other_obj.get_bbox_points(
            trans=other_obj_pos, rot=other_obj_quat
        )

        face_normals = [
            obj_points[1] - obj_points[0],
            obj_points[2] - obj_points[0],
            obj_points[3] - obj_points[0],
            other_obj_points[1] - other_obj_points[0],
            other_obj_points[2] - other_obj_points[0],
            other_obj_points[3] - other_obj_points[0],
        ]

        intersect = True

        # noramlize length of normals
        for normal in face_normals:
            normal = np.array(normal) / np.linalg.norm(normal)

            obj_projs = [np.dot(p, normal) for p in obj_points]
            other_obj_projs = [np.dot(p, normal) for p in other_obj_points]

            # see if gap detected
            if np.min(other_obj_projs) > np.max(obj_projs) or np.min(
                obj_projs
            ) > np.max(other_obj_projs):
                intersect = False
                break
    else:
        """
        old code from placement_samplers.py
        """
        obj_x, obj_y, obj_z = obj_pos
        other_obj_x, other_obj_y, other_obj_z = other_obj_pos
        xy_collision = (
            np.linalg.norm((obj_x - other_obj_x, obj_y - other_obj_y))
            <= other_obj.horizontal_radius + obj.horizontal_radius
        )
        if obj_z > other_obj_z:
            z_collision = (
                obj_z - other_obj_z <= other_obj.top_offset[-1] - obj.bottom_offset[-1]
            )
        else:
            z_collision = (
                other_obj_z - obj_z <= obj.top_offset[-1] - other_obj.bottom_offset[-1]
            )

        if xy_collision and z_collision:
            intersect = True
        else:
            intersect = False

    return intersect
