"""
Common utilities often reused for internal code and task building for users.
"""

from collections import defaultdict
from typing import Dict, Sequence, Union

import gymnasium as gym
import numpy as np
import sapien.physx as physx
import torch

from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.types import Array, Device


# -------------------------------------------------------------------------- #
# Basic
# -------------------------------------------------------------------------- #
def dict_merge(dct: dict, merge_dct: dict):
    """In place recursive merge of `merge_dct` into `dct`"""
    for k, v in merge_dct.items():
        if (
            k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], dict)
        ):  # noqa
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]


# TODO (stao): Consolidate this function with the one above..
def merge_dicts(ds: Sequence[Dict], asarray=False):
    """Merge multiple dicts with the same keys to a single one."""
    # NOTE(jigu): To be compatible with generator, we only iterate once.
    ret = defaultdict(list)
    for d in ds:
        for k in d:
            ret[k].append(d[k])
    ret = dict(ret)
    # Sanity check (length)
    assert len(set(len(v) for v in ret.values())) == 1, "Keys are not same."
    if asarray:
        ret = {k: np.concatenate(v) for k, v in ret.items()}
    return ret


def append_dict_array(
    x1: Union[dict, Sequence, Array], x2: Union[dict, Sequence, Array]
):
    """Append `x2` in front of `x1` and returns the result. Tries to do this in place if possible.
    Assumes both `x1, x2` have the same dictionary structure if they are dictionaries.
    They may also both be lists/sequences in which case this is just appending like normal"""
    if isinstance(x1, np.ndarray):
        if len(x1.shape) > len(x2.shape):
            # if different dims, check if extra dim is just a 1 due to single env in batch mode and if so, add it to x2.
            if x1.shape[1] == 1:
                x2 = x2[:, None, :]
            elif x1.shape[0] == 1:
                x2 = x2[None, ...]
        return np.concatenate([x1, x2])
    elif isinstance(x1, list):
        return x1 + x2
    elif isinstance(x1, dict):
        for k in x1.keys():
            assert k in x2, "dct and append_dct need to have the same dictionary layout"
            x1[k] = append_dict_array(x1[k], x2[k])
    return x1


def index_dict_array(x1, idx: Union[int, slice], inplace=True):
    """Indexes every array in x1 with slice and returns result."""
    if (
        isinstance(x1, np.ndarray)
        or isinstance(x1, list)
        or isinstance(x1, torch.tensor)
    ):
        return x1[idx]
    elif isinstance(x1, dict):
        if inplace:
            for k in x1.keys():
                x1[k] = index_dict_array(x1[k], idx, inplace=inplace)
            return x1
        else:
            out = dict()
            for k in x1.keys():
                out[k] = index_dict_array(x1[k], idx, inplace=inplace)
            return out


def normalize_vector(x: torch.Tensor, eps=1e-6):
    """normalizes a given torch tensor x and if the norm is less than eps, set the norm to 0"""
    norm = torch.linalg.norm(x, axis=1)
    norm[norm < eps] = 1
    norm = 1 / norm
    return torch.multiply(x, norm[:, None])


def compute_angle_between(x1: torch.Tensor, x2: torch.Tensor):
    """Compute angle (radian) between two torch tensors"""
    x1, x2 = normalize_vector(x1), normalize_vector(x2)
    dot_prod = torch.clip(torch.einsum("ij,ij->i", x1, x2), -1, 1)
    return torch.arccos(dot_prod)


# -------------------------------------------------------------------------- #
# Numpy
# -------------------------------------------------------------------------- #
def np_normalize_vector(x, eps=1e-6):
    """normalizes a given numpy array x and if the norm is less than eps, set the norm to 0"""
    x = np.asarray(x)
    assert x.ndim == 1, x.ndim
    norm = np.linalg.norm(x)
    return np.zeros_like(x) if norm < eps else (x / norm)


def np_compute_angle_between(x1: np.ndarray, x2: np.ndarray):
    """Compute angle (radian) between two numpy arrays"""
    x1, x2 = np_normalize_vector(x1), np_normalize_vector(x2)
    dot_prod = np.clip(np.dot(x1, x2), -1, 1)
    return np.arccos(dot_prod).item()


# ---------------------------------------------------------------------------- #
# OpenAI gym
# ---------------------------------------------------------------------------- #
# TODO (stao): Clean up this code
def flatten_state_dict(
    state_dict: dict, use_torch=False, device: Device = None
) -> Array:
    """Flatten a dictionary containing states recursively. Expects all data to be either torch or numpy

    Args:
        state_dict: a dictionary containing scalars or 1-dim vectors.

    Raises:
        AssertionError: If a value of @state_dict is an ndarray with ndim > 2.

    Returns:
        np.ndarray | torch.Tensor: flattened states.

    Notes:
        The input is recommended to be ordered (e.g. OrderedDict).
        However, since python 3.7, dictionary order is guaranteed to be insertion order.
    """
    states = []

    for key, value in state_dict.items():
        if isinstance(value, dict):
            state = flatten_state_dict(value, use_torch=use_torch)
            if state.size == 0:
                state = None
            if use_torch:
                state = sapien_utils.to_tensor(state)
        elif isinstance(value, (tuple, list)):
            state = None if len(value) == 0 else value
            if use_torch:
                state = sapien_utils.to_tensor(state)
        elif isinstance(value, (bool, np.bool_, int, np.int32, np.int64)):
            # x = np.array(1) > 0 is np.bool_ instead of ndarray
            state = int(value)
            if use_torch:
                state = sapien_utils.to_tensor(state)
        elif isinstance(value, (float, np.float32, np.float64)):
            state = np.float32(value)
            if use_torch:
                state = sapien_utils.to_tensor(state)
        elif isinstance(value, np.ndarray):
            if value.ndim > 2:
                raise AssertionError(
                    "The dimension of {} should not be more than 2.".format(key)
                )
            state = value if value.size > 0 else None
            if use_torch:
                state = sapien_utils.to_tensor(state)

        else:
            is_torch_tensor = False
            if isinstance(value, torch.Tensor):
                state = value
                if len(state.shape) == 1:
                    state = state[:, None]
                is_torch_tensor = True
            if not is_torch_tensor:
                raise TypeError("Unsupported type: {}".format(type(value)))
        if state is not None:
            states.append(state)

    if use_torch:
        if len(states) == 0:
            return torch.empty(0, device=device)
        else:
            return torch.hstack(states)
    else:
        if len(states) == 0:
            return np.empty(0)
        else:
            return np.hstack(states)


def flatten_dict_keys(d: dict, prefix=""):
    """Flatten a dict by expanding its keys recursively."""
    out = dict()
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten_dict_keys(v, prefix + k + "/"))
        else:
            out[prefix + k] = v
    return out
