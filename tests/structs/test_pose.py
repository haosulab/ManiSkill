import numpy as np
import pytest
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.utils.structs.pose import Pose


def test_pose_creation():
    pose = Pose.create_from_pq()
    assert pose.raw_pose.shape == (1, 7)
    assert isinstance(pose.raw_pose, torch.Tensor)
    assert torch.isclose(pose.raw_pose, torch.tensor([[0, 0, 0, 1.0, 0, 0, 0]])).all()


def test_pose_create_with_p():
    pose = Pose.create_from_pq(p=[1, 0, 2])
    assert torch.isclose(pose.raw_pose, torch.tensor([[1, 0, 2, 1.0, 0, 0, 0]])).all()

    pose = Pose.create_from_pq(p=[[1, 0, 2], [1, 0, -2]])
    assert torch.isclose(
        pose.raw_pose, torch.tensor([[1, 0, 2, 1.0, 0, 0, 0], [1, 0, -2, 1.0, 0, 0, 0]])
    ).all()


def test_pose_create_with_q():
    w, x, y, z = euler2quat(0.3, 0.4, -0.2)
    pose = Pose.create_from_pq(q=[w, x, y, z])
    assert torch.isclose(
        pose.raw_pose, torch.tensor([[0, 0, 0, w, x, y, z]]).float()
    ).all()

    w, x, y, z = euler2quat(0.3, 0.4, -0.2)
    w2, x2, y2, z2 = euler2quat(0.3, 0.4, -0.2)
    pose = Pose.create_from_pq(p=[1, 2, 3], q=[[w, x, y, z], [w2, x2, y2, z2]])
    assert torch.isclose(
        pose.raw_pose,
        torch.tensor([[1, 2, 3, w, x, y, z], [1, 2, 3, w2, x2, y2, z2]]).float(),
    ).all()


def test_pose_to_sapien_pose():
    p = [1, 2, 4]
    q = euler2quat(0, -0.3, 1)
    sapien_pose = sapien.Pose(p=p, q=q)
    pose = Pose.create_from_pq(p=p, q=q)
    assert isinstance(pose.sp, sapien.Pose)
    assert np.all(pose.sp.p == sapien_pose.p)
    assert np.all(pose.sp.q == sapien_pose.q)


def test_pose_mult():
    p = [1, 2, 4]
    q = euler2quat(0, -0.3, 1)
    sapien_pose = sapien.Pose(p=p, q=q)
    pose = Pose.create_from_pq(p=p, q=q)

    pose = pose * pose
    sapien_pose = sapien_pose * sapien_pose
    assert np.isclose(pose.sp.p, sapien_pose.p).all()
    assert np.isclose(pose.sp.q, sapien_pose.q).all()

    p = [[1, 2, 3], [-2.5, 3, 0]]
    q = [euler2quat(0, -0.3, 1), euler2quat(0.9, 0.3, -1)]
    sapien_pose1 = sapien.Pose(p=p[0], q=q[0])
    sapien_pose2 = sapien.Pose(p=p[1], q=q[1])
    pose = Pose.create_from_pq(p=p, q=q)
    pose = pose * pose
    sapien_pose1 = sapien_pose1 * sapien_pose1
    sapien_pose2 = sapien_pose2 * sapien_pose2
    for i, sp in enumerate([sapien_pose1, sapien_pose2]):
        assert np.isclose(pose.p[i], sp.p).all()
        assert np.isclose(pose.q[i], sp.q).all()


def test_pose_inv():
    p = [1, 2, 4]
    q = euler2quat(0, -0.3, 1)
    sapien_pose = sapien.Pose(p=p, q=q)
    pose = Pose.create_from_pq(p=p, q=q)

    pose = pose.inv()
    sapien_pose = sapien_pose.inv()
    assert np.isclose(pose.sp.p, sapien_pose.p).all()
    assert np.isclose(pose.sp.q, sapien_pose.q).all()

    p = [[1, 2, 3], [-2.5, 3, 0]]
    q = [euler2quat(0, -0.3, 1), euler2quat(0.9, 0.3, -1)]
    sapien_pose1 = sapien.Pose(p=p[0], q=q[0])
    sapien_pose2 = sapien.Pose(p=p[1], q=q[1])
    pose = Pose.create_from_pq(p=p, q=q)
    pose = pose.inv()
    sapien_pose1 = sapien_pose1.inv()
    sapien_pose2 = sapien_pose2.inv()
    for i, sp in enumerate([sapien_pose1, sapien_pose2]):
        assert np.isclose(pose.p[i], sp.p).all()
        assert np.isclose(pose.q[i], sp.q).all()


def test_pose_transformation_matrix():
    p = [1, 2, 4]
    q = euler2quat(0, -0.3, 1)
    sapien_pose = sapien.Pose(p=p, q=q)
    pose = Pose.create_from_pq(p=p, q=q)

    pose = pose.to_transformation_matrix()
    sapien_pose = sapien_pose.to_transformation_matrix()
    assert np.isclose(pose, sapien_pose).all()

    p = [[1, 2, 3], [-2.5, 3, 0]]
    q = [euler2quat(0, -0.3, 1), euler2quat(0.9, 0.3, -1)]
    sapien_pose1 = sapien.Pose(p=p[0], q=q[0])
    sapien_pose2 = sapien.Pose(p=p[1], q=q[1])
    pose = Pose.create_from_pq(p=p, q=q)
    pose = pose.to_transformation_matrix()
    sapien_pose1 = sapien_pose1.to_transformation_matrix()
    sapien_pose2 = sapien_pose2.to_transformation_matrix()
    for i, sp in enumerate([sapien_pose1, sapien_pose2]):
        assert np.isclose(pose[i], sp).all()
