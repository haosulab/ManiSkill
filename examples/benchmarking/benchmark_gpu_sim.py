# py-spy record -f speedscope -r 1000 -o profile -- python manualtest/benchmark_gpu_sim.py
# python manualtest/benchmark_orbit_sim.py --task "Isaac-Lift-Cube-Franka-v0" --num_envs 512 --headless
import argparse
import time
import gymnasium as gym
import sapien
import torch
import mani_skill2.envs
import sapien.physx
import tqdm
import numpy as np
import sapien.render
import sapien.physx
from mani_skill2.utils.visualization.misc import images_to_video, tile_images

def main(args):
    num_envs = args.num_envs

    sapien.physx.set_gpu_memory_config(found_lost_pairs_capacity=2**26, max_rigid_patch_count=120000)
    env = gym.make(args.env_id, num_envs=num_envs, obs_mode=args.obs_mode, enable_shadow=True, render_mode=args.render_mode, control_mode="pd_joint_delta_pos", sim_freq=100, control_freq=50)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    images = []
    video_nrows=int(np.sqrt(num_envs))
    with torch.inference_mode():
        env.reset(seed=2022)
        env.step(env.action_space.sample()) # warmup? it seems first step here is slow for some reason
        env.reset(seed=2022)
        if args.save_video:
            images.append(env.render())
        N = 100
        stime = time.time()
        for i in tqdm.tqdm(range(N)):
            actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            obs, rew, terminated, truncated, info = env.step(actions)
            if args.save_video:
                images.append(env.render())
        dtime = time.time() - stime
        FPS = num_envs * N / dtime
        print(f"{FPS=:0.3f}. {N=} frames in {dtime:0.3f}s with {num_envs} parallel envs")

        if args.save_video:
            images = [tile_images(rgbs, nrows=video_nrows).cpu().numpy() for rgbs in images]
            images_to_video(images, output_dir="./videos/benchmark", video_name=f"mani_skill_gpu_sim-num_envs={num_envs}-obs_mode={args.obs_mode}-render_mode={args.render_mode}", fps=30)
            del images
        env.reset(seed=2022)
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
    env.close()
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v0")
    parser.add_argument("-o", "--obs-mode", type=str, default="none")
    parser.add_argument("-n", "--num-envs", type=int, default=256)
    parser.add_argument(
        "--render-mode", type=str, default="rgb_array"
    ),
    parser.add_argument(
        "--save-video", action="store_true", help="whether to save videos"
    )
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    main(parse_args())
