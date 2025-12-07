"""
Smallest enclosing cylinder computation.

Based on the algorithm from:
https://www.nayuki.io/page/smallest-enclosing-circle
"""

import math
from typing import Optional, Tuple

import numpy as np


def _compute_smallest_circle(
    points: list[Tuple[float, float]]
) -> Optional[Tuple[float, float, float]]:
    points = [(float(x), float(y)) for x, y in points]
    np.random.shuffle(points)

    circle = None
    for i, point in enumerate(points):
        if circle is None or not _point_in_circle(circle, point):
            circle = _compute_circle_with_point(points[: i + 1], point)
    return circle


def _compute_circle_with_point(
    points: list[Tuple[float, float]], p: Tuple[float, float]
) -> Tuple[float, float, float]:
    circle = (p[0], p[1], 0.0)
    for i, q in enumerate(points):
        if not _point_in_circle(circle, q):
            if circle[2] == 0.0:
                circle = _get_circle_from_diameter(p, q)
            else:
                circle = _compute_circle_with_two_points(points[: i + 1], p, q)
    return circle


def _compute_circle_with_two_points(
    points: list[Tuple[float, float]], p: Tuple[float, float], q: Tuple[float, float]
) -> Tuple[float, float, float]:
    circle = _get_circle_from_diameter(p, q)
    left = right = None

    for r in points:
        if _point_in_circle(circle, r):
            continue

        cross = _compute_cross_product(p[0], p[1], q[0], q[1], r[0], r[1])
        candidate = _compute_circumcircle(p, q, r)

        if candidate is None:
            continue
        elif cross > 0 and (
            left is None
            or _compute_cross_product(
                p[0], p[1], q[0], q[1], candidate[0], candidate[1]
            )
            > _compute_cross_product(p[0], p[1], q[0], q[1], left[0], left[1])
        ):
            left = candidate
        elif cross < 0 and (
            right is None
            or _compute_cross_product(
                p[0], p[1], q[0], q[1], candidate[0], candidate[1]
            )
            < _compute_cross_product(p[0], p[1], q[0], q[1], right[0], right[1])
        ):
            right = candidate

    if left is None and right is None:
        return circle
    elif left is None:
        return right
    elif right is None:
        return left
    else:
        return left if left[2] <= right[2] else right


def _get_circle_from_diameter(
    a: Tuple[float, float], b: Tuple[float, float]
) -> Tuple[float, float, float]:
    center_x = (a[0] + b[0]) / 2
    center_y = (a[1] + b[1]) / 2
    radius = max(
        math.hypot(center_x - a[0], center_y - a[1]),
        math.hypot(center_x - b[0], center_y - b[1]),
    )
    return (center_x, center_y, radius)


def _compute_circumcircle(
    a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]
) -> Optional[Tuple[float, float, float]]:
    # Center point
    mid_x = (min(a[0], b[0], c[0]) + max(a[0], b[0], c[0])) / 2
    mid_y = (min(a[1], b[1], c[1]) + max(a[1], b[1], c[1])) / 2

    ax, ay = a[0] - mid_x, a[1] - mid_y
    bx, by = b[0] - mid_x, b[1] - mid_y
    cx, cy = c[0] - mid_x, c[1] - mid_y

    det = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2
    if det == 0:
        return None

    center_x = (
        mid_x
        + (
            (ax * ax + ay * ay) * (by - cy)
            + (bx * bx + by * by) * (cy - ay)
            + (cx * cx + cy * cy) * (ay - by)
        )
        / det
    )
    center_y = (
        mid_y
        + (
            (ax * ax + ay * ay) * (cx - bx)
            + (bx * bx + by * by) * (ax - cx)
            + (cx * cx + cy * cy) * (bx - ax)
        )
        / det
    )
    radius = max(math.hypot(center_x - p[0], center_y - p[1]) for p in (a, b, c))

    return (center_x, center_y, radius)


def _point_in_circle(
    circle: Optional[Tuple[float, float, float]], point: Tuple[float, float]
) -> bool:
    if circle is None:
        return False
    return math.hypot(point[0] - circle[0], point[1] - circle[1]) <= circle[2] * (
        1 + 1e-14
    )


def _compute_cross_product(
    x0: float, y0: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)


def aabc(points: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Compute axis-aligned bounding cylinder for 3D points.

    Args:
        points: Nx3 array of points

    Returns:
        (center_x, center_y, radius, min_z, max_z) tuple
    """
    points = np.asarray(points)
    z_bounds = points[:, 2].min(), points[:, 2].max()
    center_x, center_y, radius = _compute_smallest_circle(points[:, :2])
    return center_x, center_y, radius, z_bounds[0], z_bounds[1]
