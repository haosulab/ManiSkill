import argparse

import gymnasium as gym
import mani_skill.envs

from mani_skill.utils.wrappers import RecordEpisode

import subprocess as sp
import time

def run_visual_ppo_with_varied_sim_params(task, name, render_quality):
    print("Running visual-based PPO (varied simulation parameters)")

    start_time = time.time()

    sp.run([
        "python", 
        "examples/baselines/ppo/ppo_rgb_custom.py", 
        f"--env_id={task}", 
        f"--exp-name={name}",
        f"--num_envs={64}",
        f"--update_epochs={8}", 
        f"--num_minibatches={16}",
        f"--total_timesteps={300_000}",
        f"--eval_freq={10}",
        f"--num-steps={20}",
        f"--sim_quality={render_quality}",
        f"--vary_sim_parameters"
    ])

    end_time = time.time()
    elapsed_sec = (end_time - start_time)
    print(f"Elasped (seconds): {elapsed_sec}")

    print("End")

def run_default_visual_ppo(task, name, render_quality):
    print("Running visual-based PPO (default)")

    start_time = time.time()

    sp.run([
        "python", 
        "examples/baselines/ppo/ppo_rgb.py", 
        f"--env_id={task}", 
        f"--exp-name={name}",
        f"--num_envs={64}",
        f"--update_epochs={8}", 
        f"--num_minibatches={16}",
        f"--total_timesteps={300_000}",
        f"--eval_freq={10}",
        f"--num-steps={20}",
        f"--sim_quality={render_quality}"
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
        tasks = [
            "PushCube-v1", 
            # "PickCube-v1", 
            # "StackCube-v1", 
            # "PegInsertionSide-v1", 
            # "AssemblingKits-v1", # realsense
            # "PlugCharger-v1"
        ]

        RENDER_QUALITY = "high"
        MODALITY = "rgb"
        EXPERIMENT = "sim-params-default" # modality or sim-quality etc.
        names = [
            f"{EXPERIMENT}-{MODALITY}-pushcube-{RENDER_QUALITY}", 
            # f"modality-{MODALITY}-pickcube-{RENDER_QUALITY}",
            # f"modality-{MODALITY}-stackcube-{RENDER_QUALITY}",
            # f"modality-{MODALITY}-peginsertionside-{RENDER_QUALITY}",
            # f"modality-{MODALITY}-assemblingkits-{RENDER_QUALITY}",
            # f"modality-{MODALITY}-plugcharger-{RENDER_QUALITY}", 
        ]

        assert len(tasks) == len(names), "equal number of params"

        print(f"Render quality is {RENDER_QUALITY}")

        for idx in range(len(tasks)):
            task_name = tasks[idx]
            name = names[idx]
            print(f"Running experiment for {task_name}-{name}")
            run_default_visual_ppo(task=task_name, name=name, render_quality=RENDER_QUALITY)
            wait_time_s = 3
            print(f"Waiting for {wait_time_s} seconds ...")
            time.sleep(wait_time_s)
    elif args.visual_vary_sim:
        tasks = [
            "PushCube-v1", 
            # "PickCube-v1", 
            # "StackCube-v1", 
            # "PegInsertionSide-v1", 
            # "AssemblingKits-v1", # realsense
            # "PlugCharger-v1"
        ]

        RENDER_QUALITY = "high"
        MODALITY = "rgb"
        EXPERIMENT = "sim-params-vary" # modality or sim-quality etc.
        names = [
            f"{EXPERIMENT}-{MODALITY}-pushcube-{RENDER_QUALITY}", 
            # f"modality-{MODALITY}-pickcube-{RENDER_QUALITY}",
            # f"modality-{MODALITY}-stackcube-{RENDER_QUALITY}",
            # f"modality-{MODALITY}-peginsertionside-{RENDER_QUALITY}",
            # f"modality-{MODALITY}-assemblingkits-{RENDER_QUALITY}",
            # f"modality-{MODALITY}-plugcharger-{RENDER_QUALITY}", 
        ]

        assert len(tasks) == len(names), "equal number of params"

        print(f"Render quality is {RENDER_QUALITY}")

        for idx in range(len(tasks)):
            task_name = tasks[idx]
            name = names[idx]
            print(f"Running experiment for {task_name}-{name}")
            run_visual_ppo_with_varied_sim_params(task=task_name, name=name, render_quality=RENDER_QUALITY)
            wait_time_s = 3
            print(f"Waiting for {wait_time_s} seconds ...")
            time.sleep(wait_time_s)
    else:
        print("No option has been selected")