"""various gymnasium/gym utilities used in ManiSkill, mostly to handle observation/action spaces and noramlization"""


from typing import Dict

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from mani_skill.utils.logging_utils import logger
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


def find_max_episode_steps_value(env):
    """Finds the max episode steps parameter given by user or registered in the environment.

    This is a useful utility as not all specs may include max episode steps and some wrappers
    may need access to this in order to implement e.g. TimeLimits correctly on the GPU sim."""
    cur = env
    if isinstance(cur, gym.vector.SyncVectorEnv):
        cur = env.envs[0]
    elif isinstance(cur, gym.vector.AsyncVectorEnv):
        raise NotImplementedError(
            "Currently cannot get max episode steps of an environment wrapped with gym.vector.AsyncVectorEnv"
        )
    elif isinstance(cur, ManiSkillVectorEnv):
        cur = env._env
    while cur is not None:
        try:
            return cur.get_wrapper_attr("max_episode_steps")
        except AttributeError:
            pass
        try:
            return cur.get_wrapper_attr("_max_episode_steps")
        except AttributeError:
            pass
        if cur.spec is not None and cur.spec.max_episode_steps is not None:
            return cur.spec.max_episode_steps
        if hasattr(cur, "env"):
            cur = cur.env
        else:
            cur = None
    return None


def extract_scalars_from_info(
    info: dict, blacklist=(), batch_size=1
) -> Dict[str, float]:
    """Recursively extract scalar metrics from an info dict returned by env.step.

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
        elif batch_size == 1 and np.size(v) == 1 and not isinstance(v, str):
            try:
                ret[k] = float(v)
            except:
                pass
        elif batch_size > 1 and np.size(v) == batch_size and not isinstance(v, str):
            try:
                ret[k] = [float(v_i) for v_i in v]
            except:
                pass
    return ret


def inv_scale_action(action, low, high):
    """Inverse of `clip_and_scale_action` without clipping."""
    return (action - 0.5 * (high + low)) / (0.5 * (high - low))


# TODO (stao): this is dead code, remove?
def inv_clip_and_scale_action(action, low, high):
    """Inverse of `clip_and_scale_action`."""
    low, high = np.asarray(low), np.asarray(high)
    action = (action - 0.5 * (high + low)) / (0.5 * (high - low))
    return np.clip(action, -1.0, 1.0)


def clip_and_scale_action(action, low, high):
    """Clip action to [-1, 1] and scale according to a range [low, high]."""
    action = torch.clip(action, -1, 1)
    return 0.5 * (high + low) + 0.5 * (high - low) * action


def normalize_action_space(action_space: spaces.Box):
    assert isinstance(action_space, spaces.Box), type(action_space)
    return spaces.Box(-1, 1, shape=action_space.shape, dtype=action_space.dtype)


def get_dtype_bounds(dtype: np.dtype):
    """Gets the min and max values of a given numpy type"""
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
