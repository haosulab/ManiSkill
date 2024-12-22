import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated
import gymnasium as gym
import numpy as np
import torch
import tqdm
import tyro

import mani_skill.envs
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.benchmarking.profiling import Profiler
from mani_skill.utils.visualization.misc import images_to_video, tile_images
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
import mani_skill.examples.benchmarking.envs # import benchmark env code

BENCHMARK_ENVS = ["FrankaPickCubeBenchmark-v1", "CartpoleBalanceBenchmark-v1", "FrankaMoveBenchmark-v1"]
@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PickCube-v1"
    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "state"
    control_mode: Annotated[str, tyro.conf.arg(aliases=["-c"])] = "pd_joint_delta_pos"
    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1024
    cpu_sim: bool = False
    """Whether to use the CPU or GPU simulation"""
    seed: int = 0
    save_example_image: bool = False
    control_freq: int | None = 60
    sim_freq: int | None = 120
    num_cams: int | None = None
    """Number of cameras. Only used by benchmark environments"""
    cam_width: int | None = None
    """Width of cameras. Only used by benchmark environments"""
    cam_height: int | None = None
    """Height of cameras. Only used by benchmark environments"""
    render_mode: str = "rgb_array"
    """Which set of cameras/sensors to render for video saving. 'cameras' value will save a video showing all sensor/camera data in the observation, e.g. rgb and depth. 'rgb_array' value will show a higher quality render of the environment running."""
    save_video: bool = False
    """Whether to save videos"""
    save_results: str | None = None
    """Path to save results to. Should be path/to/results.csv"""
def main(args: Args):
    profiler = Profiler(output_format="stdout")
    num_envs = args.num_envs
    sim_config = dict()
    if args.control_freq:
        sim_config["control_freq"] = args.control_freq
    if args.sim_freq:
        sim_config["sim_freq"] = args.sim_freq
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
            render_mode=args.render_mode,
            control_mode=args.control_mode,
            sim_config=sim_config,
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
                output_dir="./videos/ms3_benchmark",
                video_name=f"mani_skill_gpu_sim-random_actions-{args.env_id}-num_envs={num_envs}-obs_mode={args.obs_mode}-render_mode={args.render_mode}",
                fps=30,
            )
            del images

        # if environment has some predefined actions run those
        if hasattr(env.unwrapped, "fixed_trajectory"):
            for k, v in env.unwrapped.fixed_trajectory.items():
                obs, _ = env.reset()
                env.step(torch.zeros(env.action_space.shape, device=base_env.device))
                obs, _ = env.reset()
                if args.save_video:
                    images = []
                    images.append(env.render().cpu().numpy())
                actions = v["actions"]
                if v["control_mode"] == "pd_joint_pos":
                    env.unwrapped.agent.set_control_mode(v["control_mode"])
                    env.unwrapped.agent.controller.reset()
                    N = v["shake_steps"] if "shake_steps" in v else 0
                    N += sum([a[1] for a in actions])
                    with profiler.profile(f"{k}_env.step", total_steps=N, num_envs=num_envs):
                        i = 0
                        for action in actions:
                            for _ in range(action[1]):
                                env.step(torch.tile(action[0], (num_envs, 1)))
                                i += 1
                                if args.save_video:
                                    images.append(env.render().cpu().numpy())
                        # runs a "shake" test, typically used to check stability of contacts/grasping
                        if "shake_steps" in v:
                            env.unwrapped.agent.set_control_mode("pd_joint_target_delta_pos")
                            env.unwrapped.agent.controller.reset()
                            while i < N:
                                actions = v["shake_action_fn"]()
                                env.step(actions)
                                if args.save_video:
                                    images.append(env.render().cpu().numpy())
                                i += 1
                    profiler.log_stats(f"{k}_env.step")
                    if args.save_video:
                        images = [tile_images(rgbs, nrows=video_nrows) for rgbs in images]
                        images_to_video(
                            images,
                            output_dir="./videos/ms3_benchmark",
                            video_name=f"mani_skill_gpu_sim-fixed_trajectory={k}-{args.env_id}-num_envs={num_envs}-obs_mode={args.obs_mode}-render_mode={args.render_mode}",
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
        if args.save_example_image:
            obs, _ = env.reset(seed=2022)
            import matplotlib.pyplot as plt
            for cam_name, cam_data in obs["sensor_data"].items():
                for k, v in cam_data.items():
                    imgs = v.cpu().numpy()
                    imgs = tile_images(imgs, nrows=int(np.sqrt(args.num_envs)))
                    cmap = None
                    if k == "depth":
                        imgs[imgs == np.inf] = 0
                        imgs = imgs[ :, :, 0]
                        cmap = "gray"
                    plt.imsave(f"maniskill_{cam_name}_{k}.png", imgs, cmap=cmap)

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
            data.update(
                num_cameras=args.num_cams,
                camera_width=args.cam_width,
                camera_height=args.cam_height,
            )
            profiler.update_csv(
                args.save_results,
                data,
            )
        except:
            pass

if __name__ == "__main__":
    main(tyro.cli(Args))
