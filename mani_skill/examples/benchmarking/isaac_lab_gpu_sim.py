# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path
import sys

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
parser.add_argument("--num-cams", type=int, default=None, help="Number of cameras. Only used by benchmark environments")
parser.add_argument("--cam-width", type=int, default=None, help="Width of cameras. Only used by benchmark environments")
parser.add_argument("--cam-height", type=int, default=None, help="Height of cameras. Only used by benchmark environments")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
assert args_cli.obs_mode != "rgbd", "IsaacLab currently does not support rendering RGB + Depth"

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
import torch
import sched, time


env_created = False
from threading import Timer

class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer     = None
        self.interval   = interval
        self.function   = function
        self.args       = args
        self.kwargs     = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False

def main():
    global env_created

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    profiler = Profiler(output_format="stdout")


    # create isaac environment
    if args_cli.obs_mode in ["rgb", "rgbd", "depth"]:
        env = gym.make(args_cli.task, cfg=env_cfg, camera_width=args_cli.cam_width, camera_height=args_cli.cam_height, num_cameras=args_cli.num_cams, obs_mode=args_cli.obs_mode, render_mode="rgb_array" if args_cli.video else None)
    else:
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    with torch.inference_mode():
        env.reset(seed=2022)
        env_created = True
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
    env_id_mapping = {
        "Isaac-Cartpole-RGB-Camera-Direct-Benchmark-v0": "CartpoleBalanceBenchmark-v1",
        "Isaac-Cartpole-Direct-Benchmark-v0": "CartpoleBalanceBenchmark-v1",
        "Isaac-Cartpole-Direct-v0": "CartpoleBalanceBenchmark-v1"
    }

    if args_cli.obs_mode in ["rgb", "rgbd", "depth"]:
        sensor_settings_str = []
        for i in range(args_cli.num_cams):
            cam_type = "RGB" if args_cli.obs_mode == "rgb" else "Depth"
            sensor_settings_str.append(f"{cam_type}({args_cli.cam_width}x{args_cli.cam_height})")
        sensor_settings_str = ", ".join(sensor_settings_str)
    else:
        sensor_settings_str = ""
    Path("benchmark_results").mkdir(parents=True, exist_ok=True)
    profiler.update_csv(
        "benchmark_results/isaac_lab.csv",
        dict(
            env_id=env_id_mapping[args_cli.task],
            obs_mode=args_cli.obs_mode,
            num_envs=args_cli.num_envs,
            # control_mode=args.control_mode,
            num_cameras=args_cli.num_cams,
            camera_width=args_cli.cam_width,
            camera_height=args_cli.cam_height,
            sensor_settings=sensor_settings_str,
            gpu_type=torch.cuda.get_device_name()
        ),
    )
    return

if __name__ == "__main__":
    def exit_on_stall():
        global env_created
        if not env_created:
            print("Simulation not running after 30 seconds. Exiting")
            os._exit(-1)
    rt = RepeatedTimer(30, exit_on_stall)
    try:
        main()
    finally:
        # close sim app
        rt.stop()
        simulation_app.close()
