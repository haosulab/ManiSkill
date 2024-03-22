# TODO(jigu): Move to sapien_utils.py
from collections import OrderedDict
from typing import Dict, Sequence

import numpy as np
import sapien
import sapien.physx as physx
from gymnasium import spaces

from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.articulation import Articulation


def get_active_joint_indices(articulation: Articulation, joint_names: Sequence[str]):
    """get the indices of the provided joint names from the Articulation's list of active joints"""
    all_joint_names = [x.name for x in articulation.get_active_joints()]
    joint_indices = [all_joint_names.index(x) for x in joint_names]
    return sapien_utils.to_tensor(joint_indices).int()


def get_joints_by_names(articulation: Articulation, joint_names: Sequence[str]):
    """Gets the Joint objects by name in the Articulation's list of active joints"""
    joints = articulation.get_active_joints()
    joint_indices = get_active_joint_indices(articulation, joint_names)
    return [joints[idx] for idx in joint_indices]


def flatten_action_spaces(action_spaces: Dict[str, spaces.Space]):
    """Flat multiple Box action spaces into a single Box space."""
    action_dims = []
    low = []
    high = []
    action_mapping = OrderedDict()
    offset = 0

    for action_name, action_space in action_spaces.items():
        if isinstance(action_space, spaces.Box):
            assert len(action_space.shape) == 1, (action_name, action_space)
        else:
            raise TypeError(action_space)

        action_dim = action_space.shape[0]
        action_dims.append(action_dim)
        low.append(action_space.low)
        high.append(action_space.high)
        action_mapping[action_name] = (offset, offset + action_dim)
        offset += action_dim

    flat_action_space = spaces.Box(
        low=np.hstack(low),
        high=np.hstack(high),
        shape=[sum(action_dims)],
        dtype=np.float32,
    )

    return flat_action_space, action_mapping
