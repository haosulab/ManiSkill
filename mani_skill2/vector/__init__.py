from .vec_env import VecEnv, VecEnvWrapper
from .wrappers.observation import (
    PointCloudObservationWrapper,
    RGBDObservationWrapper,
    RobotSegmentationObservationWrapper,
)


def make(env_id, num_envs, server_address=None, **kwargs):
    """Instantiate a vectorized ManiSkill2 environment."""
    import socket
    from functools import partial

    from mani_skill2 import logger
    from mani_skill2.utils.registration import REGISTERED_ENVS

    if env_id not in REGISTERED_ENVS:
        raise KeyError("Env {} not found in registry".format(env_id))
    env_spec = REGISTERED_ENVS[env_id]

    # Dispatch observation mode
    obs_mode = kwargs.get("obs_mode")
    if obs_mode is None:
        obs_mode = env_spec.cls.SUPPORTED_OBS_MODES[0]
    if obs_mode not in ["state", "state_dict", "none"]:
        kwargs["obs_mode"] = "image"

    # Whether to enable robot segmentation
    enable_robot_seg = kwargs.get("enable_robot_seg", False)
    if enable_robot_seg:
        camera_cfgs = kwargs.get("camera_cfgs", {})
        camera_cfgs["add_segmentation"] = True
        kwargs["camera_cfgs"] = camera_cfgs

    env_fn = partial(env_spec.make, **kwargs)

    # https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
    if server_address == "auto":
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
            server_address = f"localhost:{port}"
        logger.info("Auto server address: {}".format(server_address))

    venv = VecEnv([env_fn for _ in range(num_envs)], server_address=server_address)
    venv.obs_mode = obs_mode

    if enable_robot_seg:
        venv = RobotSegmentationObservationWrapper(venv)

    # Dispatch observation wrapper
    if obs_mode == "rgbd":
        venv = RGBDObservationWrapper(venv)
    elif obs_mode == "pointcloud":
        venv = PointCloudObservationWrapper(venv)

    return venv
