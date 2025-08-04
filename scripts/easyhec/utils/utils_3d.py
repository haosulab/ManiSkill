# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import numpy as np
import torch


def to_array(x, dtype=float):
    if isinstance(x, np.ndarray):
        return x.astype(dtype)
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(dtype)
    elif isinstance(x, list):
        return [to_array(a) for a in x]
    elif isinstance(x, dict):
        return {k: to_array(v) for k, v in x.items()}
    elif isinstance(x, TrackedArray):
        return np.array(x)
    else:
        return x


def se3_exp_map(log_transform: torch.Tensor, eps: float = 1e-4):
    from .pytorch3d_se3 import se3_exp_map

    return se3_exp_map(log_transform, eps)


def se3_log_map(
    transform: torch.Tensor,
    eps: float = 1e-4,
    cos_bound: float = 1e-4,
    backend=None,
    test_acc=True,
):
    if backend is None:
        backend = "pytorch3d"
    if backend == "pytorch3d":
        dof6 = pytorch3d.transforms.se3.se3_log_map(transform, eps, cos_bound)
    elif backend == "opencv":
        from .pytorch3d_se3 import _get_se3_V_input, _se3_V_matrix

        # from pytorch3d.common.compat import solve
        log_rotation = []
        for tsfm in transform:
            cv2_rot = -cv2.Rodrigues(to_array(tsfm[:3, :3]))[0]
            log_rotation.append(
                torch.from_numpy(cv2_rot.reshape(-1)).to(transform.device).float()
            )
        log_rotation = torch.stack(log_rotation, dim=0)
        T = transform[:, 3, :3]
        V = _se3_V_matrix(*_get_se3_V_input(log_rotation), eps=eps)
        log_translation = torch.linalg.solve(V, T[:, :, None])[:, :, 0]
        dof6 = torch.cat((log_translation, log_rotation), dim=1)
    else:
        raise NotImplementedError()
    if test_acc:
        err = (se3_exp_map(dof6) - transform).abs().max()
        if err > 0.1:
            raise RuntimeError()
    return dof6


def _se3_V_matrix(
    log_rotation: torch.Tensor,
    log_rotation_hat: torch.Tensor,
    log_rotation_hat_square: torch.Tensor,
    rotation_angles: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    A helper function that computes the "V" matrix from [1], Sec 9.4.2.
    [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    """

    V = (
        torch.eye(3, dtype=log_rotation.dtype, device=log_rotation.device)[None]
        + log_rotation_hat
        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
        * ((1 - torch.cos(rotation_angles)) / (rotation_angles**2))[:, None, None]
        + (
            log_rotation_hat_square
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            * ((rotation_angles - torch.sin(rotation_angles)) / (rotation_angles**3))[
                :, None, None
            ]
        )
    )

    return V


def _get_se3_V_input(log_rotation: torch.Tensor, eps: float = 1e-4):
    """
    A helper function that computes the input variables to the `_se3_V_matrix`
    function.
    """
    # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
    nrms = (log_rotation**2).sum(-1)
    rotation_angles = torch.clamp(nrms, eps).sqrt()
    log_rotation_hat = hat(log_rotation)
    log_rotation_hat_square = torch.bmm(log_rotation_hat, log_rotation_hat)
    return log_rotation, log_rotation_hat, log_rotation_hat_square, rotation_angles
