"""
Utils for working with ManiSkill trajectory files
"""


import h5py
import numpy as np

from mani_skill.utils.structs.types import Array

# TODO (stao): some functions here may better be moved to the common module


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
    raise NotImplementedError()
