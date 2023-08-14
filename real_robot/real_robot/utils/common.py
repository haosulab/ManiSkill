from collections import defaultdict, OrderedDict
from typing import Dict, Sequence

import numpy as np
from sapien.core import Pose
from gym import spaces

from .logger import logger


# -------------------------------------------------------------------------- #
# Basic
# -------------------------------------------------------------------------- #
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


# -------------------------------------------------------------------------- #
# Numpy
# -------------------------------------------------------------------------- #
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
def convert_observation_to_space(observation, prefix=""):
    """Convert observation to OpenAI gym observation space (recursively).
    Modified from `gym.envs.mujoco_env`
    """
    if isinstance(observation, (dict)):
        # CATUION: Explicitly create a list of key-value tuples
        # Otherwise, spaces.Dict will sort keys if a dict is provided
        space = spaces.Dict(
            [
                (k, convert_observation_to_space(v, prefix + "/" + k))
                for k, v in observation.items()
            ]
        )
    elif isinstance(observation, np.ndarray):
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


def clip_and_scale_action(action, new_range, old_range=(-1, 1)):
    """Clip action to old_range and scale according to a new_range.
    :param action: float or np.ndarray
    :param new_range: (low, high) or np.ndarray ([:, 0] is low, [:, 1] is high)
    :param old_range: (low, high) or np.ndarray ([:, 0] is low, [:, 1] is high)
    """
    new_low, new_high = np.asarray(new_range).T
    old_low, old_high = np.asarray(old_range).T

    action = np.clip(action, old_low, old_high)

    return (
        (action - old_low) / (old_high - old_low)
        * (new_high - new_low) + new_low
    )


def vectorize_pose(pose: Pose) -> np.ndarray:
    return np.hstack([pose.p, pose.q])


def flatten_state_dict(state_dict: dict) -> np.ndarray:
    """Flatten a dictionary containing states recursively.

    Args:
        state_dict: a dictionary containing scalars or 1-dim vectors.

    Raises:
        AssertionError: If a value of @state_dict is an ndarray with ndim > 2.

    Returns:
        np.ndarray: flattened states.

    Notes:
        The input is recommended to be ordered (e.g. OrderedDict).
        However, since python 3.7, dictionary order is guaranteed to be insertion order.
    """
    states = []
    for key, value in state_dict.items():
        if isinstance(value, dict):
            state = flatten_state_dict(value)
            if state.size == 0:
                state = None
        elif isinstance(value, (tuple, list)):
            state = None if len(value) == 0 else value
        elif isinstance(value, (bool, np.bool_, int, np.int32, np.int64)):
            # x = np.array(1) > 0 is np.bool_ instead of ndarray
            state = int(value)
        elif isinstance(value, (float, np.float32, np.float64)):
            state = np.float32(value)
        elif isinstance(value, np.ndarray):
            if value.ndim > 2:
                raise AssertionError(
                    f"The dimension of {key} should not be more than 2."
                )
            state = value if value.size > 0 else None
        else:
            raise TypeError("Unsupported type: {}".format(type(value)))
        if state is not None:
            states.append(state)
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


def flatten_dict_space_keys(space: spaces.Dict, prefix="") -> spaces.Dict:
    """Flatten a dict of spaces by expanding its keys recursively."""
    out = OrderedDict()
    for k, v in space.spaces.items():
        if isinstance(v, spaces.Dict):
            out.update(flatten_dict_space_keys(v, prefix + k + "/").spaces)
        else:
            out[prefix + k] = v
    return spaces.Dict(out)
