from collections import OrderedDict, defaultdict
from typing import Dict, Sequence, Union

import gymnasium as gym
import numpy as np
import sapien.physx as physx
import torch
from gymnasium import spaces

from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.types import Array, Device

from .logging_utils import logger


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
    if isinstance(x1, np.ndarray) or isinstance(x1, list):
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


def normalize_vector(x, eps=1e-6):
    norm = torch.linalg.norm(x, axis=1)
    norm[norm < eps] = 1
    norm = 1 / norm
    return torch.multiply(x, norm[:, None])


def compute_angle_between(x1, x2):
    """Compute angle (radian) between two vectors."""
    x1, x2 = normalize_vector(x1), normalize_vector(x2)
    dot_prod = torch.clip(torch.einsum("ij,ij->i", x1, x2), -1, 1)
    return torch.arccos(dot_prod)


# -------------------------------------------------------------------------- #
# Numpy
# -------------------------------------------------------------------------- #
def np_normalize_vector(x, eps=1e-6):
    x = np.asarray(x)
    assert x.ndim == 1, x.ndim
    norm = np.linalg.norm(x)
    return np.zeros_like(x) if norm < eps else (x / norm)


def np_compute_angle_between(x1, x2):
    """Compute angle (radian) between two vectors."""
    x1, x2 = np_normalize_vector(x1), np_normalize_vector(x2)
    dot_prod = np.clip(np.dot(x1, x2), -1, 1)
    return np.arccos(dot_prod).item()


def get_dtype_bounds(dtype: np.dtype):
    if np.issubdtype(dtype, np.floating):
        info = np.finfo(dtype)
        return info.min, info.max
    elif np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return info.min, info.max
    elif np.issubdtype(dtype, np.bool_):
        return 0, 1
    else:
        raise TypeError(dtype)


# ---------------------------------------------------------------------------- #
# OpenAI gym
# ---------------------------------------------------------------------------- #
def convert_observation_to_space(observation, prefix="", unbatched=False):
    """Convert observation to OpenAI gym observation space (recursively).
    Modified from `gym.envs.mujoco_env`
    """
    if isinstance(observation, (dict)):
        # CATUION: Explicitly create a list of key-value tuples
        # Otherwise, spaces.Dict will sort keys if a dict is provided
        space = spaces.Dict(
            [
                (
                    k,
                    convert_observation_to_space(
                        v, prefix + "/" + k, unbatched=unbatched
                    ),
                )
                for k, v in observation.items()
            ]
        )
    elif isinstance(observation, np.ndarray):
        if unbatched:
            shape = observation.shape[1:]
        else:
            shape = observation.shape
        dtype = observation.dtype
        low, high = get_dtype_bounds(dtype)
        if np.issubdtype(dtype, np.floating):
            low, high = -np.inf, np.inf
        space = spaces.Box(low, high, shape=shape, dtype=dtype)
    elif isinstance(observation, (float, np.float32, np.float64)):
        logger.debug(f"The observation ({prefix}) is a (float) scalar")
        space = spaces.Box(-np.inf, np.inf, shape=[1], dtype=np.float32)
    elif isinstance(observation, (int, np.int32, np.int64)):
        logger.debug(f"The observation ({prefix}) is a (integer) scalar")
        space = spaces.Box(-np.inf, np.inf, shape=[1], dtype=int)
    elif isinstance(observation, (bool, np.bool_)):
        logger.debug(f"The observation ({prefix}) is a (bool) scalar")
        space = spaces.Box(0, 1, shape=[1], dtype=np.bool_)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


def normalize_action_space(action_space: spaces.Box):
    assert isinstance(action_space, spaces.Box), type(action_space)
    return spaces.Box(-1, 1, shape=action_space.shape, dtype=action_space.dtype)


def clip_and_scale_action(action, low, high):
    """Clip action to [-1, 1] and scale according to a range [low, high]."""
    action = torch.clip(action, -1, 1)
    return 0.5 * (high + low) + 0.5 * (high - low) * action


def inv_clip_and_scale_action(action, low, high):
    """Inverse of `clip_and_scale_action`."""
    low, high = np.asarray(low), np.asarray(high)
    action = (action - 0.5 * (high + low)) / (0.5 * (high - low))
    return np.clip(action, -1.0, 1.0)


def inv_scale_action(action, low, high):
    """Inverse of `clip_and_scale_action` without clipping."""
    low, high = np.asarray(low), np.asarray(high)
    return (action - 0.5 * (high + low)) / (0.5 * (high - low))


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


def extract_scalars_from_info(info: dict, blacklist=()) -> Dict[str, float]:
    """Recursively extract scalar metrics from info dict.

    Args:
        info (dict): info dict
        blacklist (tuple, optional): keys to exclude.

    Returns:
        Dict[str, float]: scalar metrics
    """
    ret = {}
    for k, v in info.items():
        if k in blacklist:
            continue

        # Ignore placeholder
        if v is None:
            continue

        # Recursively extract scalars
        elif isinstance(v, dict):
            ret2 = extract_scalars_from_info(v, blacklist=blacklist)
            ret2 = {f"{k}.{k2}": v2 for k2, v2 in ret2.items()}
            ret2 = {k2: v2 for k2, v2 in ret2.items() if k2 not in blacklist}

        # Things that are scalar-like will have an np.size of 1.
        # Strings also have an np.size of 1, so explicitly ban those
        elif np.size(v) == 1 and not isinstance(v, str):
            ret[k] = float(v)
    return ret


def flatten_dict_space_keys(space: spaces.Dict, prefix="") -> spaces.Dict:
    """Flatten a dict of spaces by expanding its keys recursively."""
    out = OrderedDict()
    for k, v in space.spaces.items():
        if isinstance(v, spaces.Dict):
            out.update(flatten_dict_space_keys(v, prefix + k + "/").spaces)
        else:
            out[prefix + k] = v
    return spaces.Dict(out)


def find_max_episode_steps_value(env):
    cur = env
    while cur is not None:
        if hasattr(cur, "max_episode_steps"):
            return cur.max_episode_steps
        if cur.spec is not None and cur.spec.max_episode_steps is not None:
            return cur.spec.max_episode_steps
        if hasattr(cur, "env"):
            cur = env.env
        else:
            cur = None
    return None
