from typing import Annotated
import genesis as gs
import torch
import numpy as np
import gymnasium as gym
from envs.genesis.franka import FrankaBenchmarkEnv
########################## init ##########################
gs.init(backend=gs.gpu, logging_level="warning")

from dataclasses import dataclass
import tyro
from profiling import Profiler, images_to_video

@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "Genesis-Franka-Benchmark-v0"
    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "state"
    control_mode: Annotated[str, tyro.conf.arg(aliases=["-c"])] = "pd_joint_delta_pos"
    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1024
    cpu_sim: bool = False
    save_example_image: bool = False
    control_freq: int | None = 60
    sim_freq: int | None = 120
    num_cams: int | None = None
    cam_width: int | None = None
    cam_height: int | None = None
    render_mode: str = "rgb_array"
    save_video: bool = False
    save_results: str | None = None

def main(args: Args):
    profiler = Profiler(output_format="stdout")
    num_envs = args.num_envs
    env = gym.make(args.env_id, num_envs=num_envs, sim_freq=args.sim_freq, control_freq=args.control_freq, render_mode=args.render_mode)

    obs, _ = env.reset()
    N = 100
    if args.save_video:
        images = [env.unwrapped.render_rgb_array()]
    with torch.inference_mode():
        with profiler.profile("env.step", total_steps=N, num_envs=num_envs):
            for i in range(N):
                actions = (
                    2 * torch.rand(env.action_space.shape, device=gs.device)
                    - 1
                )
                obs, _, _, _, _ = env.step(actions)
                if args.save_video:
                    rgb = env.unwrapped.render_rgb_array()
                    images.append(rgb)
        env.close()
        profiler.log_stats("env.step")
        if args.save_video:
            images_to_video(images, output_dir="videos", video_name="genesis_franka_benchmark", fps=30)

if __name__ == "__main__":
    parsed_args = tyro.cli(Args)
    main(parsed_args)
