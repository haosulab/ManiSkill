from dataclasses import dataclass

# from mani_skill.utils import common, io_utils, wrappers
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro

from mani_skill.utils.visualization.misc import tile_images
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper


# TODO (xhin): support args to allow list of .png output filenames to save backgrounds to
@dataclass
class Args:
    robot_yaml_path: str = ""
    """path for the lerobot yaml config file of robot"""
    sim_env_id: str = "GrabCube-v1"
    real_env_id: str = "RealGrabCube-v1"
    keyframe_id: str = None
    """robot keyframe for task initial robot qpos"""
    debug: bool = False


def overlay_envs(sim_env, real_env):
    """
    Overlays sim_env observtions onto real_env observations
    Requires matching ids between the two environments' sensors
    e.g. id=phone_camera sensor in real_env / real_robot config, must have identical id in sim_env
    """
    real_obs = real_env.get_obs()["sensor_data"]
    sim_obs = sim_env.get_obs()["sensor_data"]
    assert sorted(real_obs.keys()) == sorted(
        sim_obs.keys()
    ), f"real camera names {real_obs.keys()} and sim camera names {sim_obs.keys()} differ"
    green_screens = dict()
    for name in real_obs:
        green_screens[name] = real_obs[name]["rgb"]

    overlaid_dict = sim_env.get_obs(alt_imgs=green_screens)["sensor_data"]
    overlaid_imgs = []
    for name in overlaid_dict:
        overlaid_imgs.append(overlaid_dict[name]["rgb"][0] / 255)

    return tile_images(overlaid_imgs)


if __name__ == "__main__":
    args = args = tyro.cli(Args)

    # set up environments
    sim_env = gym.make(
        args.sim_env_id,
        num_envs=1,
        obs_mode="rgb+segmentation",
        render_mode="rgb_array",
        debug=args.debug,
    )
    sim_env = FlattenRGBDObservationWrapper(sim_env, rgb=True, depth=False, state=True)
    sim_obs, _ = sim_env.reset(seed=0)

    real_env = gym.make(
        args.real_env_id,
        robot_yaml_path=args.robot_yaml_path,
        keyframe_id=args.keyframe_id,
        control_freq=sim_env.control_freq,
        control_mode=sim_env.control_mode,
        control_timing=not args.debug,
    )
    real_env = FlattenRGBDObservationWrapper(
        real_env, rgb=True, depth=False, state=True
    )
    obs, _ = real_env.reset()

    # for plotting robot camera reads
    fig = plt.figure()
    ax = fig.add_subplot()
    im = ax.imshow(sim_env.render()[0] / 255)
    while True:
        overlaid_imgs = overlay_envs(sim_env, real_env)
        im.set_data(overlaid_imgs)
        # Redraw the plot
        fig.canvas.draw()
        fig.show()
        fig.canvas.flush_events()
        print("Press Enter to update overlay")
        input()
