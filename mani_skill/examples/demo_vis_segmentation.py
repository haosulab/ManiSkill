import signal

from mani_skill.utils import common
signal.signal(signal.SIGINT, signal.SIG_DFL) # allow ctrl+c when using plt.show

import argparse

import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import Camera
from mani_skill.utils.structs import Actor, Link
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PushCube-v1", help="The environment ID of the task you want to simulate")
    parser.add_argument("--id", type=str, help="The ID or name of actor you want to segment and render")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of environments to run. Used for some basic testing and not visualized")
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
    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode="rgbd",
        num_envs=args.num_envs
    )

    obs, _ = env.reset(seed=args.seed)
    selected_id = args.id
    if selected_id is not None and selected_id.isdigit():
        selected_id = int(selected_id)

    n_cams = 0
    for config in env.unwrapped._sensors.values():
        if isinstance(config, Camera):
            n_cams += 1
    print(f"Visualizing {n_cams} RGBD cameras")

    print("ID to Actor/Link name mappings")
    print("0: Background")

    reverse_seg_id_map = dict()
    for obj_id, obj in sorted(env.unwrapped.segmentation_id_map.items()):
        if isinstance(obj, Actor):
            print(f"{obj_id}: Actor, name - {obj.name}")
        elif isinstance(obj, Link):
            print(f"{obj_id}: Link, name - {obj.name}")
        reverse_seg_id_map[obj.name] = obj_id
    if selected_id is not None and not isinstance(selected_id, int):
        selected_id = reverse_seg_id_map[selected_id]


    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        fig, axes = plt.subplots(nrows=n_cams, ncols=2)
        axes = axes.reshape(n_cams, 2)
        cam_num = 0

        for cam in obs["sensor_data"].keys():
            if "rgb" in obs["sensor_data"][cam]:

                rgb = common.to_numpy(obs["sensor_data"][cam]["rgb"])
                seg = common.to_numpy(obs["sensor_data"][cam]["segmentation"])
                if selected_id is not None:
                    seg = seg == selected_id
                if args.num_envs > 1:
                    rgb = rgb[0]
                    seg = seg[0]

                axes[cam_num, 0].imshow(rgb)
                axes[cam_num, 1].imshow(seg)
                axes[cam_num, 0].axis('off')
                axes[cam_num, 1].axis('off')
                cam_num += 1
        fig.tight_layout()
        plt.show()

if __name__ == "__main__":
    main(parse_args())
