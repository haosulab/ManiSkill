import argparse

import gymnasium as gym
import numpy as np

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.wrappers.record import RecordEpisode
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PushCube-v1", help="The environment ID of the task you want to simulate")
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="Can be 'human' to open a viewer, or rgb_array / sensors which change the cameras saved videos use")
    parser.add_argument("--record-dir", type=str, default="videos/reset_distributions", help="Where to save recorded videos. If none, no videos are saved")
    parser.add_argument("-n", "--num-resets", type=int, default=20, help="Number of times to reset the environment")
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Seed the random actions and environment. Default is no seed",
    )
    args = parser.parse_args()
    return args


def main(args):
    if args.seed is not None:
        np.random.seed(args.seed)
    env: BaseEnv = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode="none",
        reward_mode="none",
        render_mode=args.render_mode,
        shader_dir=args.shader,
        sim_backend=args.sim_backend,
    )
    if args.record_dir is not None and args.render_mode != "human":
        # we are not saving video via the wrapper as it does not save empty trajectories
        env = RecordEpisode(env, output_dir=args.record_dir, save_video=False, save_trajectory=False, video_fps=10)
    env.reset(seed=args.seed)

    if args.render_mode == "human":
        viewer = env.render()
        print("Rendering reset distribution in GUI. Press 'r' to reset and 'q' to quit")
        while True:
            viewer = env.render_human()
            if viewer.window.key_press("r"):
                env.reset()
            elif viewer.window.key_press("q"):
                break
    else:
        for _ in range(args.num_resets):
            env.reset()
            env.render_images.append(env.capture_image())
        name = f"{args.env_id}_reset_distribution"
        env.flush_video(name=name)
        print(f"Saved video to {env.output_dir}/{name}.mp4")
    env.close()

if __name__ == "__main__":
    main(parse_args())
