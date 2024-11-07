from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill.utils import common
from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_apply,
    quaternion_multiply,
    quaternion_to_matrix,
)
from mani_skill.utils.structs.types import Array, Device


def add_batch_dim(x):
    if len(x.shape) == 1:
        return x[None, :]
    return x


def to_batched_tensor(x: Union[List, Array], device: Optional[Device] = None):
    if x is None:
        return None
    return add_batch_dim(common.to_tensor(x, device=device))


@dataclass
class Pose:
    """
    Wrapper around sapien.Pose that supports managing a batch of Poses and flexible creation of them from a variety of
    sources (list, numpy array, sapien.Pose). This pose object will also return information with a batch dimension, even if it is just holding
    a single position and quaternion.

    As a result pose.p and pose.q will return shapes (N, 3) and (N, 4) respectively for N poses being stored. pose.raw_pose stores all the pose data as a single
    2D array of shape (N, 7).

    All sapien.Pose API are re-implemented in batch mode here to support GPU simulation. E.g. pose multiplication and inverse with ``pose_1.inv() * pose_2``,
    or creating transformation matrices with ``pose_1.to_transformation_matrix()`` are suppported they same way they are in sapien.Pose.

    Pose Creation
    -------------

    To create a batched pose with a given position ``p`` and/or quaternion ``q``, you run:

    .. code-block:: python

        pose = Pose.create_from_pq(p=p, q=q)

    ``p`` and ``q`` can be a torch tensor, numpy array, and/or list, or None.

    If ``p`` or ``q`` have only 1 value/not batched, then we automatically repeat the value to the batch size of the other given value.
    For example, if ``p`` has a batch dimension of size > 1, and ``q`` has a batch dimension of size 1 or is a flat list, then the
    code automatically repeats the ``q`` value to the batch size of ``p``. Likewise in the reverse direction the same repeating occurs.

    If ``p`` and ``q`` have the same batch size, they are stored as so.

    If ``p`` and ``q`` have no batch dimensions, one is automatically added (e.g. ``p`` having shape (3,) now becomes (1, 3))

    If ``p`` is None, it is auto filled with zeros.

    If ``q`` is None, it is auto filled with the [1, 0, 0, 0] quaternion.

    If you have a sapien.Pose, another Pose object, or a raw pose tensor of shape (N, 7) or (7,) called ``x``, you can create this Pose object with:

    .. code-block:: python

        pose = Pose.create(x)

    If you want a sapien.Pose object instead of this batched Pose, you can do ``pose.sp`` to get the sapien.Pose version (which is not batched). Note that
    this is only permitted if this Pose has a batch size of 1.

    Pose Indexing
    -------------

    You can index into a Pose object like numpy/torch arrays to get a new Pose object with the indexed data.

    For example if ``pose`` has a batch size of 4, then ``pose[0]`` will be a Pose object with batch size of 1, and
    ``pose[1:3]`` will be a Pose object with batch size of 2.

    """

    raw_pose: torch.Tensor

    @classmethod
    def create_from_pq(
        cls,
        p: Optional[torch.Tensor] = None,
        q: Optional[torch.Tensor] = None,
        device: Optional[Device] = None,
    ):
        """Creates a Pose object from a given position ``p`` and/or quaternion ``q``"""
        if device is None:
            # try to guess which device to use
            device = (
                p.device
                if p is not None and isinstance(p, torch.Tensor)
                else q.device
                if q is not None and isinstance(q, torch.Tensor)
                else None
            )
        if p is None:
            p = torch.zeros((1, 3), device=device)
        if q is None:
            q = torch.zeros((1, 4), device=device)
            q[:, 0] = 1
        p, q = to_batched_tensor(p, device=device), to_batched_tensor(q, device=device)

        # correct batch sizes if needed
        if p.shape[0] > q.shape[0]:
            assert q.shape[0] == 1
            q = q.repeat(p.shape[0], 1)
        elif p.shape[0] < q.shape[0]:
            assert p.shape[0] == 1
            p = p.repeat(q.shape[0], 1)
        raw_pose = torch.hstack([p, q])
        return cls(raw_pose=raw_pose)

    @classmethod
    def create(
        cls,
        pose: Union[torch.Tensor, sapien.Pose, List[sapien.Pose], "Pose"],
        device: Optional[Device] = None,
    ) -> "Pose":
        """Creates a Pose object from a given ``pose``, which can be a torch tensor, sapien.Pose, list of sapien.Pose, or Pose"""
        if isinstance(pose, sapien.Pose):
            raw_pose = torch.hstack(
                [
                    common.to_tensor(pose.p, device=device),
                    common.to_tensor(pose.q, device=device),
                ]
            )
            return cls(raw_pose=add_batch_dim(raw_pose))
        elif isinstance(pose, cls):
            return cls(raw_pose=pose.raw_pose.to(device))
        elif isinstance(pose, list) and isinstance(pose[0], sapien.Pose):
            ps = []
            qs = []
            for p in pose:
                ps.append(p.p)
                qs.append(p.q)
            ps = common.to_tensor(ps, device=device)
            qs = common.to_tensor(qs, device=device)
            return cls(raw_pose=torch.hstack([ps, qs]))

        else:
            assert len(pose.shape) <= 2 and len(pose.shape) > 0
            pose = common.to_tensor(pose, device=device)
            pose = add_batch_dim(pose)
            if pose.shape[-1] == 3:
                return cls.create_from_pq(p=pose, device=pose.device)
            assert pose.shape[-1] == 7
            return cls(raw_pose=pose)

    def __getitem__(self, i):
        return Pose.create(self.raw_pose[i])

    def __len__(self):
        return len(self.raw_pose)

    @property
    def shape(self) -> torch.Size:
        """Shape of the Pose object"""
        return self.raw_pose.shape

    @property
    def device(self) -> Device:
        """Torch Device the Pose object is on"""
        return self.raw_pose.device

    def to(self, device: Device):
        """Move the Pose object to a different device"""
        if self.raw_pose.device == device:
            return self
        return Pose.create(self.raw_pose.to(device))

    # -------------------------------------------------------------------------- #
    # Functions from sapien.Pose
    # -------------------------------------------------------------------------- #
    # def __getstate__(self) -> tuple: ...
    # @typing.overload
    # def __init__(self, p: numpy.ndarray[numpy.float32, _Shape, _Shape[3]] = array([0., 0., 0.], dtype=float32), q: numpy.ndarray[numpy.float32, _Shape, _Shape[4]] = array([1., 0., 0., 0.], dtype=float32)) -> None: ...
    # @typing.overload
    # def __init__(self, arg0: numpy.ndarray[numpy.float32, _Shape[4, 4]]) -> None: ...
    def __mul__(self, arg0: Union["Pose", sapien.Pose]) -> "Pose":
        """
        Multiply two poses. Supports multiplying singular poses like sapien.Pose or Pose object with batch size of 1 with Pose objects with batch size > 1.
        """
        # NOTE (stao): this code is probably slower than SAPIEN's pose multiplication but it is batched
        arg0 = Pose.create(arg0, device=self.device)
        pose = self
        if len(arg0) == 1 and len(pose) > 1:
            # repeat arg0 to match shape of self
            arg0 = Pose.create(arg0.raw_pose.repeat(len(pose), 1))
        elif len(pose) == 1 and len(arg0) > 1:
            pose = Pose.create(pose.raw_pose.repeat(len(arg0), 1))
        new_q = quaternion_multiply(pose.q, arg0.q)
        new_p = pose.p + quaternion_apply(pose.q, arg0.p)
        return Pose.create_from_pq(new_p, new_q)

    # def __repr__(self) -> str: ...
    # def __setstate__(self, arg0: tuple) -> None: ...
    def get_p(self):
        """Returns self.p, the position"""
        return self.p

    def get_q(self):
        """Returns self.q, the quaternion"""
        return self.q

    # def get_rpy(self) -> numpy.ndarray[numpy.float32, _Shape, _Shape[3]]: ...
    def inv(self):
        """Returns the inverse of this pose"""
        inverted_raw_pose = self.raw_pose.clone()
        inverted_raw_pose[..., 4:] = -inverted_raw_pose[..., 4:]
        new_p = quaternion_apply(inverted_raw_pose[..., 3:], -self.p)
        inverted_raw_pose[..., :3] = new_p
        return Pose.create(inverted_raw_pose)

    def set_p(self, p: torch.Tensor) -> None:
        """Sets the position of this pose"""
        self.p = p

    def set_q(self, q: torch.Tensor) -> None:
        """Sets the quaternion of this pose"""
        self.q = q

    # def set_rpy(self, arg0: numpy.ndarray[numpy.float32, _Shape, _Shape[3]]) -> None: ...
    def to_transformation_matrix(self):
        """Returns the (N, 4, 4) shaped transformation matrix equivalent to this pose"""
        b = self.raw_pose.shape[0]
        mat = torch.zeros((b, 4, 4), device=self.raw_pose.device)
        mat[..., :3, :3] = quaternion_to_matrix(self.q)
        mat[..., :3, 3] = self.p
        mat[..., 3, 3] = 1
        return mat

    @property
    def sp(self):
        """
        Returns the equivalent sapien pose. Note that this is only permitted if this Pose has a batch size of 1.
        """
        return to_sapien_pose(self)

    @property
    def p(self):
        """The position of this pose"""
        return self.raw_pose[..., :3]

    @p.setter
    def p(self, arg1: torch.Tensor):
        self.raw_pose[..., :3] = common.to_tensor(arg1)

    @property
    def q(self):
        """The quaternion of this pose"""
        return self.raw_pose[..., 3:]

    @q.setter
    def q(self, arg1: torch.Tensor):
        self.raw_pose[..., 3:] = common.to_tensor(arg1)

    # @property
    # def rpy(self) -> numpy.ndarray[numpy.float32, _Shape, _Shape[3]]:
    #     """
    #     :type: numpy.ndarray[numpy.float32, _Shape, _Shape[3]]
    #     """
    # @rpy.setter
    # def rpy(self, arg1: numpy.ndarray[numpy.float32, _Shape, _Shape[3]]) -> None:
    #     pass


def vectorize_pose(
    pose: Union[sapien.Pose, Pose, Array], device: Optional[Device] = None
) -> torch.Tensor:
    """
    Maps several formats of Pose representation to the appropriate tensor representation
    """
    if isinstance(pose, sapien.Pose):
        return torch.concatenate(
            [
                common.to_tensor(pose.p, device=device),
                common.to_tensor(pose.q, device=device),
            ]
        )
    elif isinstance(pose, Pose):
        return pose.raw_pose.to(device)
    else:
        return common.to_tensor(pose, device=device)


def to_sapien_pose(pose: Union[torch.Tensor, sapien.Pose, Pose]) -> sapien.Pose:
    """
    Maps several formats to a sapien Pose
    """
    if isinstance(pose, sapien.Pose):
        return pose
    elif isinstance(pose, Pose):
        pose = pose.raw_pose
        assert len(pose.shape) == 1 or (
            len(pose.shape) == 2 and pose.shape[0] == 1
        ), "pose is batched. Note that sapien Poses are not batched. If you want to use a batched Pose object use from mani_skill.utils.structs.pose import Pose"
        if len(pose.shape) == 2:
            pose = pose[0]
        pose = common.to_numpy(pose)
        return sapien.Pose(pose[:3], pose[3:])
    else:
        assert len(pose.shape) == 1 or (
            len(pose.shape) == 2 and pose.shape[0] == 1
        ), "pose is batched. Note that sapien Poses are not batched. If you want to use a batched Pose object use from mani_skill.utils.structs.pose import Pose"
        if len(pose.shape) == 2:
            pose = pose[0]
        pose = common.to_numpy(pose)
        return sapien.Pose(pose[:3], pose[3:])
