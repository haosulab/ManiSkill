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
):
    xyz_angles = torch.rand((n, 3), device=device) * torch.pi * 2
    if lock_x:
        xyz_angles[:, 0] *= 0
    if lock_y:
        xyz_angles[:, 1] *= 0
    if lock_z:
        xyz_angles[:, 2] *= 0
    return matrix_to_quaternion(euler_angles_to_matrix(xyz_angles, convention="XYZ"))
