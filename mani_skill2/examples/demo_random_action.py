import argparse

import gymnasium as gym
import numpy as np

from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.wrappers import RecordEpisode


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v0")
    parser.add_argument("-o", "--obs-mode", type=str, default="none")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-c", "--control-mode", type=str)
    parser.add_argument("--render-mode", type=str)
    parser.add_argument("--record-dir", type=str)
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output.")
    args, opts = parser.parse_known_args(args)

    # Parse env kwargs
    if not args.quiet:
        print("opts:", opts)
    eval_str = lambda x: eval(x[1:]) if x.startswith("@") else x
    env_kwargs = dict((x, eval_str(y)) for x, y in zip(opts[0::2], opts[1::2]))
    if not args.quiet:
        print("env_kwargs:", env_kwargs)
    args.env_kwargs = env_kwargs

    return args


def main(args):
    np.set_printoptions(suppress=True, precision=3)
    verbose = not args.quiet
    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        **args.env_kwargs
    )

    record_dir = args.record_dir
    if record_dir:
        record_dir = record_dir.format(env_id=args.env_id)
        env = RecordEpisode(env, record_dir)

    if verbose:
        print("Observation space", env.observation_space)
        print("Action space", env.action_space)
        print("Control mode", env.control_mode)
        print("Reward mode", env.reward_mode)

    obs, _ = env.reset()
    if args.render_mode is not None:
        env.render()

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if verbose:
            print("reward", reward)
            print("terminated", terminated)
            print("truncated", truncated)
            print("info", info)

        if args.render_mode is not None:
            env.render()

        if terminated or truncated:
            break
    env.close()


if __name__ == "__main__":
    main(parse_args())
