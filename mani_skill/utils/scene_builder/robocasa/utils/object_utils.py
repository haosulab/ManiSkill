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
    from robocasa.models.fixtures import Fixture

    if isinstance(obj, MJCFObject) or isinstance(obj, Fixture):
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
