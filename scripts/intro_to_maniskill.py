import argparse

import gymnasium as gym
import mani_skill.envs

from mani_skill.utils.wrappers import RecordEpisode

import subprocess as sp
import time

def pick_cube_intro():
    env = gym.make(
        "PickCube-v1",
        num_envs=1,
        obs_mode="state", # there is also "state_dict", "rgbd", ...
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array"
    )

    env = RecordEpisode(
        env,
        "/home/filip-grigorov/code/scripts/data/videos", # the directory to save replay videos and trajectories to
        # on GPU sim we record intervals, not by single episodes as there are multiple envs
        # each 100 steps a new video is saved
        max_steps_per_video=100
    )

    print(f"Observational space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    obs, _ = env.reset(seed=0)
    done = False

    while not done:
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        env.render()

    env.close()

def run_default_state_ppo():
    print("Running state-based PPO")

    start_time = time.time()

    sp.run([
        "python", 
        "examples/baselines/ppo/ppo.py", 
        f"--env_id={'PushCube-v1'}", 
        f"--exp-name={'state-pushcube'}",
        f"--num_envs={1024}",
        f"--update_epochs={8}", 
        f"--num_minibatches={32}",
        f"--total_timesteps={600_000}",
        f"--eval_freq={8}",
        f"--num-steps={20}"
    ])

    end_time = time.time()
    elapsed_sec = (end_time - start_time)
    print(f"Elasped (seconds): {elapsed_sec}")

    print("End")


def run_default_visual_ppo():
    print("Running visual-based PPO")

    start_time = time.time()

    sp.run([
        "python", 
        "examples/baselines/ppo/ppo_rgb.py", 
        f"--env_id={'PushCube-v1'}", 
        f"--exp-name={'rgb-pushcube'}",
        f"--num_envs={256}",
        f"--update_epochs={8}", 
        f"--num_minibatches={16}",
        f"--total_timesteps={250_000}",
        f"--eval_freq={10}",
        f"--num-steps={20}"
    ])

    end_time = time.time()
    elapsed_sec = (end_time - start_time)
    print(f"Elasped (seconds): {elapsed_sec}")

    print("End")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual", action='store_true') # default is False
    parser.add_argument("--state", action='store_true') # default is False

    args = parser.parse_args()

    if args.visual:
        run_default_visual_ppo()
    elif args.state:
        run_default_state_ppo()
    else:
        print("No option has been selected")
