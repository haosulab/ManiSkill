"""
Utils for working with ManiSkill trajectory files
"""


import h5py
import numpy as np

from mani_skill.utils.structs.types import Array


def _get_dict_len(x):
    if isinstance(x, dict) or isinstance(x, h5py.Group):
        for k in x.keys():
            return _get_dict_len(x[k])
    else:
        return len(x)


def index_dict(x, i):
    res = dict()
    if isinstance(x, dict) or isinstance(x, h5py.Group):
        for k in x.keys():
            res[k] = index_dict(x[k], i)
        return res
    else:
        return x[i]


def dict_to_list_of_dicts(x):
    result = []
    N = _get_dict_len(x)
    for i in range(N):
        result.append(index_dict(x, i))
    return result


def list_of_dicts_to_dict(x):
    """Convert a list of dictionaries into a dictionary of lists/arrays.

    This is the inverse operation of dict_to_list_of_dicts.

    Args:
        x: List of dictionaries with the same structure

    Returns:
        Dictionary where each value is a list/array containing the corresponding values from input dicts
    """
    if not x:  # Empty list
        return {}

    result = {}
    # Get keys from first dict since all should have same structure
    for key in x[0].keys():
        # If value is itself a dict, recursively convert
        if isinstance(x[0][key], dict):
            result[key] = list_of_dicts_to_dict([d[key] for d in x])
        else:
            # Convert list of values to numpy array
            result[key] = np.array([d[key] for d in x])

    return result
