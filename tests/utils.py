import numpy as np
import torch

from mani_skill2.utils.common import flatten_dict_keys

ENV_IDS = [
    "LiftCube-v0",
    "PickCube-v0",
    "StackCube-v0",
    "PickSingleYCB-v0",
    "PickSingleEGAD-v0",
    "PickClutterYCB-v0",
    "AssemblingKits-v0",
    "PegInsertionSide-v0",
    "PlugCharger-v0",
    "PandaAvoidObstacles-v0",
    "TurnFaucet-v0",
    "OpenCabinetDoor-v1",
    "OpenCabinetDrawer-v1",
    "PushChair-v1",
    "MoveBucket-v1",
]

STATIONARY_ENV_IDS = [
    "LiftCube-v0",
    "PickCube-v0",
    "StackCube-v0",
    "PickSingleYCB-v0",
    "PickSingleEGAD-v0",
    "PickClutterYCB-v0",
    "AssemblingKits-v0",
    "PegInsertionSide-v0",
    "PlugCharger-v0",
    "PandaAvoidObstacles-v0",
    "TurnFaucet-v0",
]

REWARD_MODES = ["dense", "normalized_dense", "sparse"]
CONTROL_MODES_STATIONARY_SINGLE_ARM = [
    "pd_joint_delta_pos",
    "pd_joint_pos",
    "pd_joint_vel",
    "pd_joint_pos_vel",
    "pd_ee_delta_pose",
    "pd_ee_delta_pos",
]
OBS_MODES = [
    "state_dict",
    "state",
    "rgbd",
    "pointcloud",
    "rgbd_robot_seg",
    "pointcloud_robot_seg",
]
VENV_OBS_MODES = [
    "image",
    "rgbd",
    "pointcloud",
    "rgbd_robot_seg",
    "pointcloud_robot_seg",
]
ROBOTS = ["panda", "xmate3_robotiq"]


def assert_obs_equal(obs1, obs2, ignore_col_vector_shape_mismatch=False):
    """Check if two observations are equal

    ignore_col_vector_shape_mismatch - If true, will ignore shape mismatch if one shape is (n, 1) but another is (n, ). this is added since
        SB3 outputs scalars as (n, ) whereas Gymnasium and ManiSkill2 use (n, 1)
    """
    if isinstance(obs1, dict):
        obs2 = flatten_dict_keys(obs2)
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
