from typing import Callable, List

import numpy as np
import torch

from mani_skill.utils import common
from mani_skill.utils.registration import REGISTERED_ENVS

ENV_IDS = list(REGISTERED_ENVS.keys())
ENV_IDS.remove("RoboCasaKitchen-v1")  # remove slow envs
MULTI_AGENT_ENV_IDS = ["TwoRobotStackCube-v1", "TwoRobotPickCube-v1"]

STATIONARY_ENV_IDS = [
    "PickCube-v1",
    "StackCube-v1",
    "PickSingleYCB-v1",
    "PegInsertionSide-v1",
]

REWARD_MODES = ["dense", "normalized_dense", "sparse"]
CONTROL_MODES_STATIONARY_SINGLE_ARM = [
    "pd_joint_delta_pos",
    "pd_joint_pos",
    # "pd_joint_vel",
    # "pd_joint_pos_vel",
    "pd_ee_delta_pose",
    "pd_ee_delta_pos",
]
OBS_MODES = [
    "state_dict",
    "state",
    "rgb",
    "rgb+depth+segmentation",
    "pointcloud",
    "depth+state",
    "state+rgb+segmentation"
    # "rgbd_robot_seg",
    # "pointcloud_robot_seg",
]
VENV_OBS_MODES = [
    "state",
    "rgbd",
    # "pointcloud",
    # "rgbd_robot_seg",
    # "pointcloud_robot_seg",
]
SINGLE_ARM_STATIONARY_ROBOTS = ["panda"]

LOW_MEM_SIM_CONFIG = dict(
    gpu_memory_config=dict(
        max_rigid_patch_count=81920, found_lost_pairs_capacity=262144
    )
)


def tree_map(x, func: Callable):
    if isinstance(x, dict):
        [tree_map(y, func) for y in x.values()]
    else:
        func(x)


def assert_isinstance(obs1, types: List):
    if not isinstance(types, list):
        types = [types]
    if isinstance(obs1, dict):
        [assert_isinstance(x, types) for x in obs1.values()]
    else:
        assert np.any([isinstance(obs1, x) for x in types])


def assert_obs_equal(obs1, obs2, ignore_col_vector_shape_mismatch=False):
    """Check if two observations are equal

    ignore_col_vector_shape_mismatch - If true, will ignore shape mismatch if one shape is (n, 1) but another is (n, ). this is added since
        SB3 outputs scalars as (n, ) whereas Gymnasium and ManiSkill use (n, 1)
    """
    obs1, obs2 = common.to_numpy(obs1), common.to_numpy(obs2)
    if isinstance(obs1, dict):
        assert isinstance(obs2, dict)
        obs1 = common.flatten_dict_keys(obs1)
        obs2 = common.flatten_dict_keys(obs2)
        for k, v in obs1.items():
            v2 = obs2[k]
            if isinstance(v2, torch.Tensor):
                v2 = v2.cpu().numpy()
            if v.dtype == np.uint8:
                # https://github.com/numpy/numpy/issues/19183
                np.testing.assert_allclose(
                    np.float32(v), np.float32(v2), err_msg=k, atol=1
                )
            elif np.issubdtype(v.dtype, np.integer):
                np.testing.assert_equal(v, v2, err_msg=k)
            else:
                if ignore_col_vector_shape_mismatch:
                    if v.ndim == 1 or v2.ndim == 1:
                        assert v.shape[0] == v2.shape[0]
                        v = v.reshape(-1)
                        v2 = v2.reshape(-1)
                np.testing.assert_allclose(v, v2, err_msg=k, atol=1e-4)
    else:
        np.testing.assert_allclose(obs1, obs2, atol=1e-4)
