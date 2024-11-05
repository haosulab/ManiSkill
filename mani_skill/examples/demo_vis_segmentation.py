import signal

from mani_skill.utils import common
from mani_skill.utils import visualization
signal.signal(signal.SIGINT, signal.SIG_DFL) # allow ctrl+c

import argparse

import gymnasium as gym
import numpy as np
# color pallete generated via https://medialab.github.io/iwanthue/
color_pallete = np.array([[164,74,82],
[85,200,95],
[149,88,210],
[111,185,57],
[89,112,223],
[194,181,43],
[219,116,216],
[71,146,48],
[214,70,164],
[157,183,57],
[154,68,158],
[82,196,133],
[225,64,121],
[50,141,77],
[224,59,84],
[74,201,189],
[237,93,68],
[77,188,225],
[182,58,29],
[77,137,200],
[230,155,53],
[93,90,162],
[213,106,38],
[150,153,224],
[120,134,37],
[186,135,220],
[78,110,27],
[182,61,117],
[106,184,145],
[184,62,65],
[44,144,124],
[229,140,186],
[48,106,60],
[167,102,155],
[160,187,114],
[150,74,107],
[204,177,86],
[34,106,77],
[226,129,94],
[72,106,45],
[222,125,129],
[101,146,86],
[150,89,44],
[147,138,73],
[210,156,106],
[102,96,32],
[168,124,34]]
, np.uint8)
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import Camera
from mani_skill.utils.structs import Actor, Link
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PushCube-v1", help="The environment ID of the task you want to simulate")
    parser.add_argument("--id", type=str, help="The ID or name of actor you want to segment and render")
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
        obs_mode="rgb+depth+segmentation",
        num_envs=args.num_envs,
        sensor_configs=sensor_configs
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

    renderer = visualization.ImageRenderer()
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        cam_num = 0
        imgs=[]
        for cam in obs["sensor_data"].keys():
            if "rgb" in obs["sensor_data"][cam]:

                rgb = common.to_numpy(obs["sensor_data"][cam]["rgb"][0])
                seg = common.to_numpy(obs["sensor_data"][cam]["segmentation"][0])
                if selected_id is not None:
                    seg = seg == selected_id
                imgs.append(rgb)
                seg_rgb = np.zeros_like(rgb)
                seg = seg % len(color_pallete)
                for id, color in enumerate(color_pallete):
                    seg_rgb[(seg == id)[..., 0]] = color
                imgs.append(seg_rgb)
                cam_num += 1
        img = visualization.tile_images(imgs, nrows=n_cams)
        renderer(img)

if __name__ == "__main__":
    main(parse_args())
