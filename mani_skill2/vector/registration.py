from functools import partial
from typing import Sequence

import gymnasium as gym

from .vec_env import PointCloudVecEnv, RGBDVecEnv, VecEnv
from .wrappers.observation import VecRobotSegmentationObservationWrapper


def _make_env(
    env_spec,
    wrappers: Sequence[gym.Wrapper] = None,
    max_episode_steps: int = None,
    **kwargs,
):
    env = env_spec.make(**kwargs)

    # Follow gym.make
    env.unwrapped.spec = env_spec.gym_spec
    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps)
    elif env_spec.max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=env_spec.max_episode_steps)

    # Add wrappers
    if wrappers is not None:
        for wrapper in wrappers:
            env = wrapper(env)

    return env


def make(
    env_id,
    num_envs,
    server_address="auto",
    wrappers: Sequence[gym.Wrapper] = None,
    enable_segmentation=False,
    max_episode_steps: int = None,
    **kwargs,
) -> VecEnv:
    """Instantiate vectorized ManiSkill2 environments.

    Args:
        env_id (str): Environment ID.
        num_envs (int): Number of environments.
        server_address (str, optional): The network address of the SAPIEN RenderServer.
            If "auto", the server will be created automatically at an avaiable port.
            Otherwise, it should be a networkd address, e.g. "localhost:12345".
        wrappers (Sequence[gym.Wrapper], optional): Wrappers for the individual environment.
        enable_segmentation (bool, optional): Whether to include "Segmentation" texture in observations.
        max_episode_steps (int, optional): Override any existing max_episode_steps/TimeLimitWrapper
            used in the initial specification of the environment.
        **kwargs: Keyword arguments to pass to the environment.
    """
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

    env_fn = partial(
        _make_env, env_spec, wrappers, max_episode_steps=max_episode_steps, **kwargs
    )

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
