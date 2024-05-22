import argparse

import gymnasium as gym
import mani_skill.envs

from mani_skill.utils.wrappers import RecordEpisode

import subprocess as sp
import time

def run_default_visual_ppo_with_varying_sim_params(task='PushCube-v1', name='rgb-pushcube-depth', render_quality='high'):
    print("Running visual-based PPO")

    start_time = time.time()

    sp.run([
        "python", 
        "examples/baselines/ppo/ppo_depth_custom.py", 
        f"--env_id={task}", 
        f"--exp-name={name}",
        f"--num_envs={40}",
        f"--update_epochs={8}", 
        f"--num_minibatches={16}",
        f"--total_timesteps={250_000}",
        f"--eval_freq={10}",
        f"--num-steps={20}",
        f"--sim_quality={render_quality}",
        #f"--random_cam_pose"
    ])

    end_time = time.time()
    elapsed_sec = (end_time - start_time)
    print(f"Elasped (seconds): {elapsed_sec}")

    print("End")

def run_default_visual_ppo(task='PushCube-v1', name='rgb-pushcube'):
    """ Uses rasterization """
    print("Running visual-based PPO")

    start_time = time.time()

    sp.run([
        "python", 
        "examples/baselines/ppo/ppo_rgb.py", 
        f"--env_id={task}", 
        f"--exp-name={name}",
        f"--num_envs={64}",#256, 64, 32
        f"--update_epochs={8}", 
        f"--num_minibatches={16}",
        f"--total_timesteps={250_000}",
        f"--eval_freq={10}",
        f"--num-steps={20}",
        f"--sim_quality={''}" # rasterization
    ])

    end_time = time.time()
    elapsed_sec = (end_time - start_time)
    print(f"Elasped (seconds): {elapsed_sec}")

    print("End")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual_default", action='store_true') # default is False
    parser.add_argument("--visual_depth", action='store_true') # default is False

    args = parser.parse_args()

    if args.visual_default:
        run_default_visual_ppo()

    elif args.visual_depth:
        tasks = [
            "PushCube-v1", 
            # "PickCube-v1", 
            # "StackCube-v1", 
            # "PegInsertionSide-v1", 
            # "AssemblingKits-v1", # realsense
            # "PlugCharger-v1"
        ]

        RENDER_QUALITY = "high"
        MODALITY = "depth+rgb+state" # depth+rgb+state, depth+rgb, depth
        names = [
            f"modality-{MODALITY}-pushcube-{RENDER_QUALITY}", 
            # "rgb-pickcube-high-raytracing-depth", 
            # "rgb-stackcube-high-raytracing-depth", 
            # "rgb-peginsertionside-high-raytracing-depth", 
            # "rgb-assemblingkits-high-raytracing-depth", 
            # "rgb-plugcharger-high-raytracing-depth"
        ]

        assert len(tasks) == len(names), "equal number of params"

        print(f"Render quality is {RENDER_QUALITY}")

        for idx in range(len(tasks)):
            task_name = tasks[idx]
            name = names[idx]
            print(f"Running experiment for {task_name}-{name}")
            run_default_visual_ppo_with_varying_sim_params(task=task_name, name=name, render_quality=RENDER_QUALITY)
            wait_time_s = 3
            print(f"Waiting for {wait_time_s} seconds ...")
            time.sleep(wait_time_s)
    else:
        print("No option has been selected")