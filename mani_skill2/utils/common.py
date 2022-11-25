from collections import OrderedDict
from typing import Dict, Iterable, List, Sequence

import gym
import numpy as np
from gym import spaces

from .logging import logger


# -------------------------------------------------------------------------- #
# Basic
# -------------------------------------------------------------------------- #
def validate_keys(ds: Iterable[dict]):
    """Validate whether keys are same."""
    keys = None
    for metrics in ds:
        metrics_keys = list(metrics.keys())
        if keys is None:
            keys = metrics_keys
        elif keys != metrics_keys:
            raise RuntimeError(
                "Different keys exist: {} vs. {}".format(keys, metrics_keys)
            )
    return keys


def merge_dicts(ds: Iterable[Dict], asarray=False):
    keys = validate_keys(ds)
    ret = {k: [d[k] for d in ds] for k in keys}
    if asarray:
        ret = {k: np.concatenate(v) for k, v in ret.items()}
    return ret


# -------------------------------------------------------------------------- #
# Numpy
# -------------------------------------------------------------------------- #
def normalize_vector(x, eps=1e-6):
    x = np.asarray(x)
    assert x.ndim == 1, x.ndim
    norm = np.linalg.norm(x)
    return np.zeros_like(x) if norm < eps else (x / norm)


def compute_angle_between(x1, x2):
    """Compute angle (radian) between two vectors."""
    x1, x2 = normalize_vector(x1), normalize_vector(x2)
    dot_prod = np.clip(np.dot(x1, x2), -1, 1)
    return np.arccos(dot_prod).item()


class np_random:
    """Context manager for numpy random state"""

    def __init__(self, seed):
        self.seed = seed
        self.state = None

    def __enter__(self):
        self.state = np.random.get_state()
        np.random.seed(self.seed)
        return self.state

    def __exit__(self, exc_type, exc_val, exc_tb):
        np.random.set_state(self.state)


def random_choice(x: Sequence, rng: np.random.RandomState = np.random):
    assert len(x) > 0
    if len(x) == 1:
        return x[0]
    else:
        return x[rng.randint(len(x))]


def get_dtype_bounds(dtype: np.dtype):
    if np.issubdtype(dtype, np.floating):
        info = np.finfo(dtype)
        return info.min, info.max
    elif np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return info.min, info.max
    elif np.issubdtype(dtype, np.bool):
        return 0, 1
    elif np.issctype(dtype, bool):
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
        space = spaces.Dict(
            {
                k: convert_observation_to_space(v, prefix + "/" + k)
                for k, v in observation.items()
            }
        )
    elif isinstance(observation, np.ndarray):
        shape = observation.shape
        dtype = observation.dtype
        dtype_min, dtype_max = get_dtype_bounds(dtype)
        low = np.full(shape, dtype_min)
        high = np.full(shape, dtype_max)
        space = spaces.Box(low, high, dtype=dtype)
    elif isinstance(observation, (float, np.float32)):
        logger.warning(f"The observation ({prefix}) is a float")
        space = spaces.Box(-np.inf, np.inf, shape=[1], dtype=np.float32)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


def normalize_action_space(action_space: spaces.Box):
    assert isinstance(action_space, spaces.Box), type(action_space)
    return spaces.Box(-1, 1, shape=action_space.shape, dtype=action_space.dtype)


def clip_and_scale_action(action, low, high):
    """Clip action to [-1, 1] and scale according to a range [low, high]."""
    low, high = np.asarray(low), np.asarray(high)
    action = np.clip(action, -1, 1)
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


def flatten_state_dict(state_dict: OrderedDict):
    """Flatten an ordered dict containing states recursively.

    Args:
        state_dict (OrderedDict): an ordered dict containing states.

    Raises:
        TypeError: If @state_dict is not an OrderedDict.
        TypeError: If a value of @state_dict is a dict instead of OrderedDict.
        AssertionError: If a value of @state_dict is an ndarray with ndim > 1.

    Returns:
        np.ndarray: flattened states.
    """
    # if not isinstance(state_dict, OrderedDict):
    #     raise TypeError(
    #         "Must be an OrderedDict, but received {}".format(type(state_dict))
    #     )
    if len(state_dict) == 0:
        return np.empty(0)
    states = []
    for key, value in state_dict.items():
        if isinstance(value, dict):
            states.append(flatten_state_dict(value))
        elif isinstance(value, (int, float, tuple, list, np.float32)):
            states.append(value)
        elif isinstance(value, np.ndarray):
            assert value.ndim <= 1, "Too many dimensions({}) for {}".format(
                value.ndim, key
            )
            if value.size > 0:
                states.append(value)
        elif isinstance(value, np.bool_):
            # x = np.array(1) > 0 is np.bool_ instead of ndarray
            states.append(value.astype(int))
        else:
            raise TypeError("Unsupported type: {}".format(type(value)))
    if states:
        return np.hstack(states)
    else:
        return []


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
