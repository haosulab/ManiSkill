# py-spy record -f speedscope -r 1000 -o profile -- python manualtest/benchmark_gpu_sim.py
# python manualtest/benchmark_orbit_sim.py --task "Isaac-Lift-Cube-Franka-v0" --num_envs 512 --headless
import argparse
import time
import numpy as np
import tqdm

# from mani_skill.utils.visualization.misc import images_to_video, tile_images


def main(args):
    num_envs = args.num_envs
    import robosuite as suite

    # create environment instance
    env = suite.make(
        env_name="Lift",  # try with other tasks like "Stack" and "Door"
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    print(
        "# -------------------------------------------------------------------------- #"
    )
    print(
        f"Benchmarking RoboSuite CPU Simulation with {num_envs} parallel environments"
    )
    print(
        f"env_id={args.env_id}, obs_mode={args.obs_mode}, control_mode={args.control_mode}"
    )
    print(f"render_mode={args.render_mode}, save_video={args.save_video}")
    print(
        f"sim_freq={env.unwrapped.sim_freq}, control_freq={env.unwrapped.control_freq}"
    )
    print(f"observation space: {env.observation_space}")
    print(f"action space: {env.action_space}")
    print(
        "# -------------------------------------------------------------------------- #"
    )
    images = []
    video_nrows = int(np.sqrt(num_envs))
    env.reset(seed=2022)
    env.step(env.action_space.sample())  # warmup step
    env.reset(seed=2022)
    if args.save_video:
        images.append(env.render())
    N = 100
    stime = time.time()
    for i in tqdm.tqdm(range(N)):
        obs, rew, terminated, truncated, info = env.step(actions)
        if args.save_video:
            images.append(env.render())
    dtime = time.time() - stime
    FPS = num_envs * N / dtime
    print(f"{FPS=:0.3f}. {N=} steps in {dtime:0.3f}s with {num_envs} parallel envs")

    if args.save_video:
        images = [tile_images(rgbs, nrows=video_nrows).cpu().numpy() for rgbs in images]
        images_to_video(
            images,
            output_dir="./videos/benchmark",
            video_name=f"mani_skill_gpu_sim-num_envs={num_envs}-obs_mode={args.obs_mode}-render_mode={args.render_mode}",
            fps=30,
        )
        del images
    env.reset(seed=2022)
    N = 1000
    stime = time.time()
    for i in tqdm.tqdm(range(N)):
        actions = (
            2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
        )
        obs, rew, terminated, truncated, info = env.step(actions)
        if i % 200 == 0 and i != 0:
            env.reset()
    dtime = time.time() - stime
    FPS = num_envs * N / dtime
    print(
        f"{FPS=:0.3f}. {N=} steps in {dtime:0.3f}s with {num_envs} parallel envs with step+reset"
    )
    env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="Lift-v0")
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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
