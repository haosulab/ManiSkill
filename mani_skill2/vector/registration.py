from typing import Sequence

import gym

from .vec_env import PointCloudVecEnv, RGBDVecEnv, VecEnv
from .wrappers.observation import VecRobotSegmentationObservationWrapper


def make(
    env_id,
    num_envs,
    server_address="auto",
    wrappers: Sequence[gym.Wrapper] = None,
    enable_segmentation=False,
    **kwargs,
) -> VecEnv:
    """Instantiate a vectorized ManiSkill2 environment.

    Args:
        env_id (str): Environment ID.
        num_envs (int): Number of environments.
        server_address (str, optional): Server address.
        wrappers (Sequence[gym.Wrapper], optional): Wrappers to wrap the environment.
        enable_segmentation (bool, optional): Whether to include Segmentation in observations.
        **kwargs: Keyword arguments to pass to the environment.
    """
    from functools import partial

    # Avoid circular import
    from mani_skill2.utils.registration import REGISTERED_ENVS

    if env_id not in REGISTERED_ENVS:
        raise KeyError("Env {} not found in registry".format(env_id))
    env_spec = REGISTERED_ENVS[env_id]

    # Dispatch observation mode
    obs_mode = kwargs.get("obs_mode")
    if obs_mode is None:
        obs_mode = env_spec.cls.SUPPORTED_OBS_MODES[0]
    if obs_mode not in ["state", "state_dict", "none", "particles"]:
        kwargs["obs_mode"] = "image"

    # Add segmentation texture
    if "robot_seg" in obs_mode:
        enable_segmentation = True
    if enable_segmentation:
        camera_cfgs = kwargs.get("camera_cfgs", {})
        camera_cfgs["add_segmentation"] = True
        kwargs["camera_cfgs"] = camera_cfgs

    env_fn = partial(env_spec.make, wrappers=wrappers, **kwargs)
    # Dispatch observation mode
    if "image" in obs_mode:
        venv_cls = VecEnv
    elif "rgbd" in obs_mode:
        venv_cls = RGBDVecEnv
    elif "pointcloud" in obs_mode:
        venv_cls = PointCloudVecEnv
    else:
        raise NotImplementedError(
            f"Unsupported observation mode for VecEnv: {obs_mode}"
        )
    venv = venv_cls([env_fn for _ in range(num_envs)], server_address=server_address)
    venv.obs_mode = obs_mode

    if "robot_seg" in obs_mode:
        venv = VecRobotSegmentationObservationWrapper(venv)

    return venv
