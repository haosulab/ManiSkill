# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--obs_mode", type=str, default="state", help="Observation mode")
parser.add_argument("--num-cams", type=int, default=1, help="Number of cameras. Only used by benchmark environments")
parser.add_argument("--cam-width", type=int, default=128, help="Width of cameras. Only used by benchmark environments")
parser.add_argument("--cam-height", type=int, default=128, help="Height of cameras. Only used by benchmark environments")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import os
from datetime import datetime
from profiling import Profiler

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg
import envs.isaaclab
def main():
    """Train with RL-Games agent."""
    # parse seed from command line
    args_cli_seed = args_cli.seed

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    profiler = Profiler(output_format="stdout")
    import torch
    # create isaac environment
    if args_cli.obs_mode in ["rgb", "rgbd"]:
        env = gym.make(args_cli.task, cfg=env_cfg, camera_width=args_cli.cam_width, camera_height=args_cli.cam_height, num_cameras=args_cli.num_cams, obs_mode=args_cli.obs_mode, render_mode="rgb_array" if args_cli.video else None)
    import matplotlib.pyplot as plt
    import ipdb;ipdb.set_trace()
    with torch.inference_mode():
        env.reset(seed=2022)
        env.step(torch.from_numpy(env.action_space.sample()).cuda())  # warmup step
        env.reset(seed=2022)
        torch.manual_seed(0)

        N = 1000
        with profiler.profile("env.step", total_steps=N, num_envs=args_cli.num_envs):
            for i in range(N):
                actions = (
                    2 * torch.rand(env.action_space.shape, device=env.unwrapped.device)
                    - 1
                )
                obs, rew, terminated, truncated, info = env.step(actions)
        profiler.log_stats("env.step")
        env.reset(seed=2022)
        N = 1000
        with profiler.profile("env.step+env.reset", total_steps=N, num_envs=args_cli.num_envs):
            for i in range(N):
                actions = (
                    2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                )
                obs, rew, terminated, truncated, info = env.step(actions)
                if i % 200 == 0 and i != 0:
                    env.reset()
        profiler.log_stats("env.step+env.reset")
    env.close()

    # append results to csv
    try:
        env_id_mapping = {
            "Isaac-Cartpole-RGB-Camera-Direct-Benchmark-v0": "CartpoleBalanceBenchmark-v1",
            "Isaac-Cartpole-Direct-Benchmark-v0": "CartpoleBalanceBenchmark-v1"
        }
        sensor_settings_str = []
        for uid, cam in base_env._sensors.items():
            if isinstance(cam, Camera):
                cfg = cam.cfg
                sensor_settings_str.append(f"RGBD({cfg.width}x{cfg.height})")
        profiler.update_csv(
            "benchmark_results/isaac_lab.csv",
            dict(
                env_id=env_id_mapping[args_cli.task],
                obs_mode=args_cli.obs_mode,
                num_envs=args_cli.num_envs,
                # control_mode=args.control_mode,
                sensor_settings=sensor_settings_str,
                gpu_type=torch.cuda.get_device_name()
            ),
        )
    except:
        pass
    return

if __name__ == "__main__":
    """Test with
    isaaclab -p benchmark_isaac_lab.py \
        --headless --task=Isaac-Lift-Cube-Franka-v0 --num_envs=1024
    """
    # run the main function
    main()
    # close sim app
    simulation_app.close()
