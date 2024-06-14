import argparse
from pathlib import Path
from turtle import update

import gymnasium as gym
import numpy as np
import sapien.physx
import sapien.render
import torch
import tqdm

import mani_skill.envs
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import Camera
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.examples.benchmarking.profiling import Profiler
from mani_skill.utils.visualization.misc import images_to_video, tile_images
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
import mani_skill.examples.benchmarking.envs # import benchmark env code

BENCHMARK_ENVS = ["PickCubeBenchmark-v1", "CartpoleBalanceBenchmark-v1"]

def main(args):
    profiler = Profiler(output_format="stdout")
    num_envs = args.num_envs
    sim_cfg = dict()
    if args.control_freq:
        sim_cfg["control_freq"] = args.control_freq
    if args.sim_freq:
        sim_cfg["sim_freq"] = args.sim_freq
    if not args.cpu_sim:
        kwargs = dict()
        if args.env_id in BENCHMARK_ENVS:
            kwargs = dict(
                camera_width=args.cam_width,
                camera_height=args.cam_height,
                num_cameras=args.num_cams,
            )

        env = gym.make(
            args.env_id,
            num_envs=num_envs,
            obs_mode=args.obs_mode,
            # enable_shadow=True,
            render_mode=args.render_mode,
            control_mode=args.control_mode,
            sim_cfg=sim_cfg,
            **kwargs
        )
        if isinstance(env.action_space, gym.spaces.Dict):
            env = FlattenActionSpaceWrapper(env)
        base_env: BaseEnv = env.unwrapped
    else:
        env = gym.make_vec(args.env_id, num_envs=args.num_envs, vectorization_mode="async", vector_kwargs=dict(context="spawn"), obs_mode=args.obs_mode,)
        base_env = gym.make(args.env_id, obs_mode=args.obs_mode).unwrapped

    base_env.print_sim_details()
    images = []
    video_nrows = int(np.sqrt(num_envs))
    with torch.inference_mode():
        env.reset(seed=2022)
        env.step(env.action_space.sample())  # warmup step
        env.reset(seed=2022)
        if args.save_video:
            images.append(env.render().cpu().numpy())
        N = 1000
        with profiler.profile("env.step", total_steps=N, num_envs=num_envs):
            for i in range(N):
                actions = (
                    2 * torch.rand(env.action_space.shape, device=base_env.device)
                    - 1
                )
                obs, rew, terminated, truncated, info = env.step(actions)
                if args.save_video:
                    images.append(env.render().cpu().numpy())
        profiler.log_stats("env.step")

        if args.save_video:
            images = [tile_images(rgbs, nrows=video_nrows) for rgbs in images]
            images_to_video(
                images,
                output_dir="./videos/benchmark",
                video_name=f"mani_skill_gpu_sim-{args.env_id}-num_envs={num_envs}-obs_mode={args.obs_mode}-render_mode={args.render_mode}",
                fps=30,
            )
            del images
        env.reset(seed=2022)
        N = 1000
        with profiler.profile("env.step+env.reset", total_steps=N, num_envs=num_envs):
            for i in range(N):
                actions = (
                    2 * torch.rand(env.action_space.shape, device=base_env.device) - 1
                )
                obs, rew, terminated, truncated, info = env.step(actions)
                if i % 200 == 0 and i != 0:
                    env.reset()
        profiler.log_stats("env.step+env.reset")
    env.close()
    if args.save_results:
        # append results to csv
        try:
            assert (
                args.save_video == False
            ), "Saving video slows down speed a lot and it will distort results"
            Path("benchmark_results").mkdir(parents=True, exist_ok=True)
            data = dict(
                env_id=args.env_id,
                obs_mode=args.obs_mode,
                num_envs=args.num_envs,
                control_mode=args.control_mode,
                gpu_type=torch.cuda.get_device_name()
            )
            if args.env_id in BENCHMARK_ENVS:
                data.update(
                    num_cameras=args.num_cams,
                    camera_width=args.cam_width,
                    camera_height=args.cam_height,
                )
            profiler.update_csv(
                "benchmark_results/maniskill.csv",
                data,
            )
        except:
            pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v1")
    parser.add_argument("-o", "--obs-mode", type=str, default="state")
    parser.add_argument("-c", "--control-mode", type=str, default="pd_joint_delta_pos")
    parser.add_argument("-n", "--num-envs", type=int, default=1024)
    parser.add_argument("--cpu-sim", action="store_true", help="Whether to use the CPU or GPU simulation")
    parser.add_argument("--control-freq", type=int, default=None, help="The control frequency to use")
    parser.add_argument("--sim-freq", type=int, default=None, help="The simulation frequency to use")
    parser.add_argument("--num-cams", type=int, default=None, help="Number of cameras. Only used by benchmark environments")
    parser.add_argument("--cam-width", type=int, default=None, help="Width of cameras. Only used by benchmark environments")
    parser.add_argument("--cam-height", type=int, default=None, help="Height of cameras. Only used by benchmark environments")
    parser.add_argument(
        "--render-mode",
        type=str,
        default="sensors",
        help="which set of cameras/sensors to render for video saving. 'cameras' value will save a video showing all sensor/camera data in the observation, e.g. rgb and depth. 'rgb_array' value will show a higher quality render of the environment running.",
    ),
    parser.add_argument(
        "--save-video", action="store_true", help="whether to save videos"
    )
    parser.add_argument(
        "--save-results", action="store_true", help="whether to save results to a csv file"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
