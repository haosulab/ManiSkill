import signal

from mani_skill.utils import common
from mani_skill.utils.visualization.misc import tile_images
signal.signal(signal.SIGINT, signal.SIG_DFL) # allow ctrl+c

import argparse

import gymnasium as gym
import cv2
import numpy as np

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import Camera
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PushCube-v1", help="The environment ID of the task you want to simulate")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of environments to run. Used for some basic testing and not visualized")
    parser.add_argument("--cam-width", type=int, help="Override the width of every camera in the environment")
    parser.add_argument("--cam-height", type=int, help="Override the height of every camera in the environment")
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Seed the random actions and environment. Default is no seed",
    )
    args = parser.parse_args()
    return args


def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
    sensor_configs = dict()
    if args.cam_width:
        sensor_configs["width"] = args.cam_width
    if args.cam_height:
        sensor_configs["height"] = args.cam_height
    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode="rgbd",
        num_envs=args.num_envs,
        sensor_configs=sensor_configs
    )

    obs, _ = env.reset(seed=args.seed)
    n_cams = 0
    for config in env.unwrapped._sensors.values():
        if isinstance(config, Camera):
            n_cams += 1
    print(f"Visualizing {n_cams} RGBD cameras")

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        cam_num = 0
        imgs=[]
        for cam in obs["sensor_data"].keys():
            if "rgb" in obs["sensor_data"][cam]:

                rgb = common.to_numpy(obs["sensor_data"][cam]["rgb"][0])
                depth = common.to_numpy(obs["sensor_data"][cam]["depth"][0]).astype(np.float32)
                depth = depth / (depth.max() - depth.min())
                imgs.append(rgb)
                depth_rgb = np.zeros_like(rgb)
                depth_rgb[..., :] = depth*255
                imgs.append(depth_rgb)
                cam_num += 1
        img = tile_images(imgs, nrows=n_cams)

        cv2.imshow('image',cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)


if __name__ == "__main__":
    main(parse_args())
