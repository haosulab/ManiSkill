"""
scripts:

## installation
mamba create -n "ms3-octo" "python==3.10.12"
mamba activate ms3-octo
pip install -e . # install mani_skill
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


from collections import defaultdict
import os
import signal
import sys

from matplotlib import pyplot as plt

from mani_skill.utils import common
from mani_skill.utils import visualization
from mani_skill.utils.visualization.misc import images_to_video
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
signal.signal(signal.SIGINT, signal.SIG_DFL) # allow ctrl+c

import argparse

import gymnasium as gym
import numpy as np
from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval import *
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import Camera
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PutCarrotOnPlateInScene-v1", help="The environment ID of the task you want to simulate")
    parser.add_argument("--shader", default="minimal", type=str, help="Change shader used for all cameras in the environment for rendering. Default is 'minimal' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of environments to run. Used for some basic testing and not visualized")
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
    # if args.cam_width:
    #     sensor_configs["width"] = args.cam_width
    # if args.cam_height:
    #     sensor_configs["height"] = args.cam_height
    sensor_configs["shader_pack"] = args.shader
    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode="rgb+segmentation",
        num_envs=args.num_envs,
        sensor_configs=sensor_configs,
        sim_backend="cpu",
    )
    env = CPUGymWrapper(env)

    obs, _ = env.reset(seed=args.seed)
    n_cams = 0
    for config in env.unwrapped._sensors.values():
        if isinstance(config, Camera):
            n_cams += 1
    print(f"Visualizing {n_cams} RGBD cameras")

    from simpler_env.policies.octo.octo_model import OctoInference
    # viewer = env.render_human()
    # viewer.paused=True;env.render_human()
    # while True:
    #     env.step(None)
    #     env.render_human()

    model_name = "octo-base"
    policy_setup = "widowx_bridge"
    model = OctoInference(model_type=model_name, policy_setup=policy_setup, init_rng=0)

    renderer = visualization.ImageRenderer(wait_for_button_press=False)
    def render_obs(obs):
        # obs["sensor_data"]
        # import ipdb; ipdb.set_trace()
        return common.to_numpy(obs["sensor_data"]["3rd_view_camera"]["rgb"], dtype=np.uint8)
        cam_num = 0
        imgs=[]
        for cam in obs["sensor_data"].keys():
            if "rgb" in obs["sensor_data"][cam]:
                rgb = common.to_numpy(obs["sensor_data"][cam]["rgb"], dtype=np.uint8)
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
    infos = []
    eval_metrics = defaultdict(list)
    for seed in range(args.seed, args.seed+100):
        obs, _ = env.reset(seed=seed)
        # while True:
        #     render_obs(obs)
        #     obs, _, _, _, _ = env.step(env.action_space.sample())
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
            img = render_obs(obs)
            img = visualization.put_info_on_image(img, info)
            # import ipdb; ipdb.set_trace()
            images.append(img)
        for k, v in info.items():
            eval_metrics[k].append(v)
        images_to_video(images, "videos/real2sim_eval/", f"octo_eval_{seed}", fps=10, verbose=True)
        for k, v in eval_metrics.items():
            print(f"{k}: {np.mean(v)}")

    import ipdb; ipdb.set_trace()
if __name__ == "__main__":
    main(parse_args())
