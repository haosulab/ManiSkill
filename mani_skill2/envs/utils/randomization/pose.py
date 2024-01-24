import numpy as np
import transforms3d

from mani_skill2.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
)
from mani_skill2.utils.sapien_utils import to_tensor


def random_quaternions(
    rng: np.random.RandomState,
    lock_x: bool = False,
    lock_y: bool = False,
    lock_z: bool = False,
    n=1,
):
    xyz_angles = rng.uniform(0, np.pi * 2, (n, 3))
    if lock_x:
        xyz_angles[:, 0] *= 0
    if lock_y:
        xyz_angles[:, 1] *= 0
    if lock_z:
        xyz_angles[:, 2] *= 0
    return matrix_to_quaternion(
        euler_angles_to_matrix(to_tensor(xyz_angles), convention="XYZ")
    )
