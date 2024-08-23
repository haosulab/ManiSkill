"""
scripts:

## installation
mamba create -n "ms3-octo" "python==3.10.12"
mamba activate ms3-octo
pip install -e .
pip install torch==2.3.1

git clone https://github.com/simpler-env/SimplerEnv
cd SimplerEnv
pip install -e .
pip install tensorflow==2.15.0
pip install -r requirements_full_install.txt
pip install tensorflow[and-cuda]==2.15.1 # tensorflow gpu support

pip install --upgrade "jax[cuda12_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
git clone https://github.com/octo-models/octo/
cd octo
git checkout 653c54acde686fde619855f2eac0dd6edad7116b  # we use octo-1.0
pip install -e .
"""


import os
import signal
import sys

from matplotlib import pyplot as plt

from mani_skill.utils import common
from mani_skill.utils import visualization
from mani_skill.utils.visualization.misc import images_to_video
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
        sensor_configs=sensor_configs,
    )

    obs, _ = env.reset(seed=args.seed)
    n_cams = 0
    for config in env.unwrapped._sensors.values():
        if isinstance(config, Camera):
            n_cams += 1
    print(f"Visualizing {n_cams} RGBD cameras")

    from simpler_env.policies.octo.octo_model import OctoInference


    model_name = "octo-base"
    policy_setup = "widowx_bridge"
    model = OctoInference(model_type=model_name, policy_setup=policy_setup, init_rng=0)



    renderer = visualization.ImageRenderer(wait_for_button_press=False)
    def render_obs(obs):
        cam_num = 0
        imgs=[]
        for cam in obs["sensor_data"].keys():
            if "rgb" in obs["sensor_data"][cam]:
                rgb = common.to_numpy(obs["sensor_data"][cam]["rgb"][0], dtype=np.uint8)
                imgs.append(rgb)
                if "depth" in obs["sensor_data"][cam]:
                    depth = common.to_numpy(obs["sensor_data"][cam]["depth"][0]).astype(np.float32)
                    depth = depth / (depth.max() - depth.min())
                    depth_rgb = np.zeros_like(rgb)
                    depth_rgb[..., :] = depth*255
                    imgs.append(depth_rgb)
                cam_num += 1
        img = visualization.tile_images(imgs, nrows=n_cams)
        # renderer(img)
        return img

    # gt_actions = np.load(os.path.join(os.path.dirname(__file__), "actions.npy"))
    for seed in range(100, 200):
        obs, _ = env.reset(seed=seed)
        instruction = env.unwrapped.get_language_instruction()
        print("instruction:", instruction)
        model.reset(instruction)
        images = []
        images.append(render_obs(obs))
        predicted_terminated, truncated = False, False
        while not (predicted_terminated or truncated):
            raw_action, action = model.step(images[-1])
            predicted_terminated = bool(action["terminate_episode"][0] > 0)
            action = np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
            obs, reward, terminated, truncated, info = env.step(action)
            truncated = bool(truncated)
            images.append(render_obs(obs))
        images_to_video(images, "videos/real2sim_eval/", f"octo_eval_{seed}", fps=10, verbose=True)

if __name__ == "__main__":
    main(parse_args())
