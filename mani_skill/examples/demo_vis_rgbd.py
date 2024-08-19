import signal
import sys

from matplotlib import pyplot as plt

from mani_skill.utils import common
from mani_skill.utils import visualization
signal.signal(signal.SIGINT, signal.SIG_DFL) # allow ctrl+c

import argparse

import gymnasium as gym
import numpy as np

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import Camera
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PushCube-v1", help="The environment ID of the task you want to simulate")
    parser.add_argument("-o", "--obs-mode", type=str, default="rgbd", help="Can be rgb or rgbd")
    parser.add_argument("--shader", default="minimal", type=str, help="Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
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

import matplotlib.pyplot as plt
import numpy as np





def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
    sensor_configs = dict()
    if args.cam_width:
        sensor_configs["width"] = args.cam_width
    if args.cam_height:
        sensor_configs["height"] = args.cam_height
    sensor_configs["shader_pack"] = args.shader
    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        num_envs=args.num_envs,
        sensor_configs=sensor_configs
    )

    obs, _ = env.reset(seed=args.seed)
    n_cams = 0
    for config in env.unwrapped._sensors.values():
        if isinstance(config, Camera):
            n_cams += 1
    print(f"Visualizing {n_cams} RGBD cameras")

    renderer = visualization.ImageRenderer()

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        cam_num = 0
        imgs=[]
        for cam in obs["sensor_data"].keys():
            if "rgb" in obs["sensor_data"][cam]:
                rgb = common.to_numpy(obs["sensor_data"][cam]["rgb"][0])
                imgs.append(rgb)
                if "depth" in obs["sensor_data"][cam]:
                    depth = common.to_numpy(obs["sensor_data"][cam]["depth"][0]).astype(np.float32)
                    depth = depth / (depth.max() - depth.min())
                    depth_rgb = np.zeros_like(rgb)
                    depth_rgb[..., :] = depth*255
                    imgs.append(depth_rgb)
                cam_num += 1
        img = visualization.tile_images(imgs, nrows=n_cams)
        renderer(img)

if __name__ == "__main__":
    main(parse_args())
