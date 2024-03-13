import numpy as np
import torch
import transforms3d

from mani_skill.utils import sapien_utils
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
from mani_skill.utils.structs.types import Device


def random_quaternions(
    n: int,
    device: Device = None,
    lock_x: bool = False,
    lock_y: bool = False,
    lock_z: bool = False,
    bounds=(0, np.pi * 2),
):
    """
    Generates random quaternions by generating random euler angles uniformly, with each of
    the X, Y, Z angles ranging from bounds[0] to bounds[1] radians. Can optionally
    choose to fix X, Y, and/or Z euler angles to 0 via lock_x, lock_y, lock_z arguments
    """
    dist = bounds[1] - bounds[0]
    xyz_angles = torch.rand((n, 3), device=device) * (dist) + bounds[0]
    if lock_x:
        xyz_angles[:, 0] *= 0
    if lock_y:
        xyz_angles[:, 1] *= 0
    if lock_z:
        xyz_angles[:, 2] *= 0
    return matrix_to_quaternion(euler_angles_to_matrix(xyz_angles, convention="XYZ"))
