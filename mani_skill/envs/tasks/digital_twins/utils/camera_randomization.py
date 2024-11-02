"""Create a pointcloud of a partial spherical shell, to be used as camera positions"""
import numpy as np
import torch

from mani_skill.utils.geometry.rotation_conversions import euler_angles_to_matrix


def batch_uniform_sample_on_unit_hemisphere(n, max_height, max_theta):
    """
    n number of points to sample
    max_height (0,1] sample points maximum z value
    max_theta (0, pi] angle of each horizontal slice
    """
    assert (
        0 < max_height <= 1
    ), "samples on unit hemisphere slice must have z value between 0 and 1"
    assert (
        0 < max_theta <= np.pi
    ), "samples on unit hemisphere slice must have theta value between 0 and pi"
    # sampling method augmented from https://mathworld.wolfram.com/SpherePointPicking.html

    phi = torch.arccos(torch.rand(n) * max_height)
    theta = (torch.rand(n) * max_theta) + (np.pi - max_theta) / 2
    x = phi.sin() * theta.cos()
    y = phi.sin() * theta.sin()
    z = phi.cos()

    return torch.vstack((x, y, z)).T


def make_camera_partial_spherical_shell(
    n, center, r1, r2, max_height, max_theta, z_orientation
):
    # using formula (1) X = sqrt((r2^2 - r1^2)p + r1^2) to uniform sample between these two radii
    # derived from pdf = 2pix / (pi(r1^2-r2^2)), for r1 <= x <= r2
    # cdf = p = (X^2 - r1^2) / (r2^2 - r1^2), inverse == eq (1)
    # we sample on outer circle first, then scale according to eq(1)
    unit_max_height = (max_height) / r2
    outer_sphere_points = r2 * batch_uniform_sample_on_unit_hemisphere(
        n, unit_max_height, max_theta
    )
    x = torch.sqrt((r2**2 - r1**2) * torch.rand(n) + r1**2)  # r1 < x < r2
    # outer_sphere points are of length r2 already, scale to this value of x
    radius_scale = x / r2
    points = outer_sphere_points * radius_scale.view(-1, 1)
    # orient in direction of z_orientation
    rot_mat = euler_angles_to_matrix(
        torch.tensor([0, 0, z_orientation], dtype=torch.float32), convention="XYZ"
    )
    oriented_points = points @ rot_mat.T
    transformed_points = oriented_points + torch.tensor(center, dtype=torch.float32)
    return transformed_points
