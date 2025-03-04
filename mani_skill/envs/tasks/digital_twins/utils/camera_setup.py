import json
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro

from mani_skill.utils.visualization.misc import tile_images
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper


@dataclass
class Args:
    sim_env_id: str = "GrabCube-v1"
    real_env_id: str = "RealGrabCube-v1"
    user_kwargs_path: Optional[str] = None
    """path to json with extra env kwargs"""
    output_photo_path: Optional[str] = None
    """path to save photo from real_env"""


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
    env_kwargs = dict()
    if args.user_kwargs_path is not None:
        with open(args.user_kwargs_path, "r") as f:
            user_kwargs = json.load(f)
        # add user_kwargs to env kwargs
        env_kwargs.update(user_kwargs)

    sim_env = gym.make(
        args.sim_env_id,
        num_envs=1,
        obs_mode="rgb+segmentation",
        render_mode="rgb_array",
        debug=True,  # by default, basedigitaltwinsenv debug = True for 50/50 overlay
        **env_kwargs,
    )
    sim_env = FlattenRGBDObservationWrapper(sim_env, rgb=True, depth=False, state=True)
    sim_obs, _ = sim_env.reset(seed=0)

    real_env = gym.make(
        args.real_env_id,
        control_freq=sim_env.control_freq,
        control_mode=sim_env.control_mode,
        control_timing=False,  # no stepping occurs
    )
    real_env = FlattenRGBDObservationWrapper(
        real_env, rgb=True, depth=False, state=True
    )
    real_env.reset()
    for i in range(10):
        real_env.step(None)

    # for plotting robot camera reads
    fig = plt.figure()
    ax = fig.add_subplot()
    im = ax.imshow(sim_env.render()[0] / 255)
    torch.set_printoptions(precision=4)
    if args.output_photo_path is not None:
        path, ext = args.output_photo_path.split(".")
        real_obs = real_env.get_obs()["sensor_data"]
        for i, name in enumerate(real_obs):
            plt.imsave(path + "_" + name + "." + ext, real_obs[name]["rgb"][0].numpy())
    else:
        obs, _ = real_env.reset()
        # qpos_ckpt = torch.tensor([[0.2, 2.2, 2.75, -0.25, -np.pi / 2, 1.0]])
        # real_env.agent.reset(qpos_ckpt[0])
        # sim_env.agent.robot.set_qpos(qpos_ckpt)
        print("Camera alignment: Move real camera to align, close figure to exit")
        while True:
            overlaid_imgs = overlay_envs(sim_env, real_env)
            print("diff", sim_env.agent.robot.qpos - real_env.agent.qpos)
            im.set_data(overlaid_imgs)
            # Redraw the plot
            fig.canvas.draw()
            fig.show()
            fig.canvas.flush_events()
            if not plt.fignum_exists(fig.number):
                print("The figure has been closed.")
                break
