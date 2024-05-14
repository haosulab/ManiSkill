import argparse

import gymnasium as gym
import mani_skill.envs

from mani_skill.utils.wrappers import RecordEpisode

import subprocess as sp
import time

def run_default_visual_ppo_with_varying_sim_params():
    print("Running visual-based PPO")

    start_time = time.time()

    sp.run([
        "python", 
        "examples/baselines/ppo/ppo_rgb_with_varying_sim.py", 
        f"--env_id={'PushCube-v1'}", 
        f"--exp-name={'rgb-pushcube'}",
        f"--num_envs={64}",
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

def run_default_visual_ppo():
    print("Running visual-based PPO")

    start_time = time.time()

    sp.run([
        "python", 
        "examples/baselines/ppo/ppo_rgb.py", 
        f"--env_id={'PushCube-v1'}", 
        f"--exp-name={'rgb-pushcube'}",
        f"--num_envs={64}",#256
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
    parser.add_argument("--visual_default", action='store_true') # default is False
    parser.add_argument("--visual_vary_sim", action='store_true') # default is False

    args = parser.parse_args()

    if args.visual_default:
        run_default_visual_ppo()
    elif args.visual_vary_sim:
        run_default_visual_ppo_with_varying_sim_params()
    else:
        print("No option has been selected")