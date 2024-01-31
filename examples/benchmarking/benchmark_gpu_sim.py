# py-spy record -f speedscope -r 1000 -o profile -- python manualtest/benchmark_gpu_sim.py
# python manualtest/benchmark_orbit_sim.py --task "Isaac-Lift-Cube-Franka-v0" --num_envs 512 --headless
import argparse
import time

import gymnasium as gym
import numpy as np
import sapien
import sapien.physx
import sapien.render
import torch
import tqdm

import mani_skill2.envs
from profiling import Profiler
from mani_skill2.utils.visualization.misc import images_to_video, tile_images


def main(args):
    profiler = Profiler(output_format=args.format)
    num_envs = args.num_envs
    # TODO (stao): we need to auto set this gpu memory config somehow
    sapien.physx.set_gpu_memory_config(
        found_lost_pairs_capacity=2**26, max_rigid_patch_count=2**19, max_rigid_contact_count=2**20
    )
    env = gym.make(
        args.env_id,
        num_envs=num_envs,
        obs_mode=args.obs_mode,
        # enable_shadow=True,
        render_mode=args.render_mode,
        control_mode=args.control_mode,
        sim_freq=100,
        control_freq=50,
    )
    print(
        "# -------------------------------------------------------------------------- #"
    )
    print(
        f"Benchmarking ManiSkill GPU Simulation with {num_envs} parallel environments"
    )
    print(
        f"env_id={args.env_id}, obs_mode={args.obs_mode}, control_mode={args.control_mode}"
    )
    print(f"render_mode={args.render_mode}, save_video={args.save_video}")
    print(f"sim_freq={env.unwrapped.sim_freq}, control_freq={env.unwrapped.control_freq}")
    print(f"observation space: {env.observation_space}")
    print(f"action space: {env.unwrapped.single_action_space}")
    print(
        "# -------------------------------------------------------------------------- #"
    )
    images = []
    video_nrows = int(np.sqrt(num_envs))
    with torch.inference_mode():
        env.reset(seed=2022)
        env.step(env.action_space.sample())  # warmup step
        env.reset(seed=2022)
        if args.save_video:
            images.append(env.render())
        N = 100
        with profiler.profile("env.step", total_steps=N, num_envs=num_envs):
            for i in range(N):
                actions = (
                    2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                )
                obs, rew, terminated, truncated, info = env.step(actions)
                if args.save_video:
                    images.append(env.render())
        profiler.log_stats("env.step")

        if args.save_video:
            images = [
                tile_images(rgbs, nrows=video_nrows).cpu().numpy() for rgbs in images
            ]
            images_to_video(
                images,
                output_dir="./videos/benchmark",
                video_name=f"mani_skill_gpu_sim-num_envs={num_envs}-obs_mode={args.obs_mode}-render_mode={args.render_mode}",
                fps=30,
            )
            del images
        env.reset(seed=2022)
        N = 1000
        with profiler.profile("env.step+env.reset", total_steps=N, num_envs=num_envs):
            for i in range(N):
                actions = (
                    2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                )
                obs, rew, terminated, truncated, info = env.step(actions)
                if i % 200 == 0 and i != 0:
                    env.reset()
        profiler.log_stats("env.step+env.reset")
    env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v1")
    parser.add_argument("-o", "--obs-mode", type=str, default="state")
    parser.add_argument("-c", "--control-mode", type=str, default="pd_joint_delta_pos")
    parser.add_argument("-n", "--num-envs", type=int, default=1024)
    parser.add_argument(
        "--render-mode",
        type=str,
        default="cameras",
        help="which set of cameras/sensors to render for video saving. 'cameras' value will save a video showing all sensor/camera data in the observation, e.g. rgb and depth. 'rgb_array' value will show a higher quality render of the environment running.",
    ),
    parser.add_argument(
        "--save-video", action="store_true", help="whether to save videos"
    )
    parser.add_argument(
        "-f", "--format", type=str, default="stdout", help="format of results. Can be stdout or json."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
