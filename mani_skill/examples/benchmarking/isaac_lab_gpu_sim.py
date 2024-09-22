import argparse
import sys

import numpy as np
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Benchmark Isaac Lab")
parser.add_argument("--num-envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--obs-mode", type=str, default="state", help="Observation mode")
parser.add_argument("--num-cams", type=int, default=None, help="Number of cameras. Only used by benchmark environments")
parser.add_argument("--cam-width", type=int, default=None, help="Width of cameras. Only used by benchmark environments")
parser.add_argument("--cam-height", type=int, default=None, help="Height of cameras. Only used by benchmark environments")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--save-results", action="store_true", help="whether to save results to a csv file"
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
from profiling import Profiler, tile_images
import torch
from pathlib import Path
import envs.isaaclab
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg

def main():
    profiler = Profiler(output_format="stdout")

    env_cfg = parse_env_cfg(
        args_cli.task, num_envs=args_cli.num_envs
    )
    # create isaac environment
    if args_cli.obs_mode != "state":
        env = gym.make(args_cli.task, cfg=env_cfg, camera_width=args_cli.cam_width, camera_height=args_cli.cam_height, num_cameras=args_cli.num_cams, obs_mode=args_cli.obs_mode)
    else:
        env = gym.make(args_cli.task, cfg=env_cfg)
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
    # import matplotlib.pyplot as plt
    # import ipdb;ipdb.set_trace()
    # if "rgb" in obs["sensors"]["cam_0"]:
    #     rgb_images = obs["sensors"]["cam_0"]["rgb"].cpu().numpy()
    #     plt.imsave("test.png", tile_images(rgb_images, nrows=int(np.sqrt(args_cli.num_envs))))
    # if "depth" in obs["sensors"]["cam_0"]:
    #     depth_images = obs["sensors"]["cam_0"]["depth"].cpu().numpy()
    #     depth_images = tile_images(depth_images, nrows=int(np.sqrt(args_cli.num_envs)))
    #     depth_images[depth_images == np.inf] = 0
    #     plt.imsave("depth.png", depth_images[:, :, 0])
    # tile_images()

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
        sensor_settings_str = None
    if args_cli.save_results:
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


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
