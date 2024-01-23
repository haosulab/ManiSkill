# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

from __future__ import annotations
import time

import tqdm

"""Launch Isaac Sim Simulator first."""


import argparse
import logging
import carb

logging.getLogger("omni.hydra").setLevel(logging.ERROR)
logging.getLogger("omni.isaac.urdf").setLevel(logging.ERROR)
logging.getLogger("omni.physx.plugin").setLevel(logging.ERROR)

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Orbit environments.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
import os
app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import traceback

import carb

import omni.isaac.contrib_tasks  # noqa: F401
import omni.isaac.orbit_tasks  # noqa: F401
from omni.isaac.orbit_tasks.utils import parse_env_cfg


def main():
    """Zero actions agent with Orbit environment."""
    # parse configuration
    num_envs = args_cli.num_envs
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=num_envs)
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    with torch.inference_mode():
        # reset environment
        env.reset(seed=2022)
        torch.manual_seed(0)

        N = 100
        stime = time.time()
        for i in tqdm.tqdm(range(N)):
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            obs, rew, terminated, truncated, info = env.step(actions)
        dtime = time.time() - stime
        FPS = num_envs * N / dtime
        print(f"{FPS=:0.3f}. {N=} frames in {dtime:0.3f}s with {num_envs} parallel envs")

        env.reset(seed=2022)
        torch.manual_seed(0)

        N = 1000
        stime = time.time()
        for i in tqdm.tqdm(range(N)):
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            obs, rew, terminated, truncated, info = env.step(actions)
            if i % 200 == 0 and i != 0:
                env.reset()
                print("RESET")
        dtime = time.time() - stime
        FPS = num_envs * N / dtime
        print(f"{FPS=:0.3f}. {N=} frames in {dtime:0.3f}s with {num_envs} parallel envs with step+reset")

    # close the simulator
    env.close()


if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
