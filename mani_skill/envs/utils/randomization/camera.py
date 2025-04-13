import torch

from mani_skill.utils import common, sapien_utils
from mani_skill.utils.geometry.rotation_conversions import (
    axis_angle_to_quaternion,
    euler_angles_to_matrix,
    quaternion_multiply,
)
from mani_skill.utils.structs.pose import Pose


def make_camera_rectangular_prism(
    n, scale=[0.1, 0.1, 0.1], center=[0, 0, 0], theta=0, device=None
) -> torch.Tensor:
    """
    Args:
        n: number of sampled points within the geometry
        scale: [x,y,z] scale for unit cube
        center: [x,y,z] scaled unit cube coordinates
        theta: [0,2pi] rotation about the z axis
    """
    scale = common.to_tensor(scale, device=device)
    center = common.to_tensor(center, device=device)
    xyz = (torch.rand(n, 3, device=device) - 0.5) * scale
    rot_mat = euler_angles_to_matrix(
        torch.tensor([0, 0, theta], dtype=torch.float32, device=device),
        convention="XYZ",
    )
    return (xyz @ rot_mat.T) + center


def noised_look_at(
    eye, target, look_at_noise=1e-2, view_axis_rot_noise=2e-1, device=None
) -> Pose:
    """
    Args:
        eye: mean camera position
        target: mean target position
        look_at_noise: std of noise added to target in lookat transform
        view_axis_rot_noise: std of noise added to rotation about the looking direction
    """
    eye = common.to_tensor(eye, device=device)
    target = common.to_tensor(target, device=device)
    targets = target.view(1, 3).repeat(len(eye), 1)
    noised_targets = torch.normal(mean=targets, std=look_at_noise)
    poses = sapien_utils.look_at(eye=eye, target=noised_targets, device=device)

    # axis to rotate around is the look at dirsection
    angles = torch.normal(
        torch.zeros(noised_targets.shape[0], device=device), std=view_axis_rot_noise
    )
    axes = noised_targets - eye
    unit_axes = axes / torch.linalg.norm(axes, dim=-1).view(-1, 1)
    axis_angle = unit_axes.view(-1, 3) * angles.view(-1, 1)
    assert axis_angle.shape[0] == angles.shape[0], (axis_angle.shape, angles.shape)

    # apply the rotation after the look_at rotation, then apply lookat translation
    # look_at rotation is camera to world, it is looking down what is in axes variable
    transforms = axis_angle_to_quaternion(axis_angle)
    return Pose.create_from_pq(poses.p, q=quaternion_multiply(transforms, poses.q))
