import argparse

import gymnasium as gym
import mani_skill.envs

from mani_skill.utils.wrappers import RecordEpisode

import subprocess as sp
import time


def run_default_visual_octo(task, name, render_quality):
    print("Running visual-based Octo")

    start_time = time.time()

    sp.run([
        "python", 
        "examples/baselines/octo/octo_rgb_eval.py", 
        f"--env_id={task}", 
        f"--exp-name={name}",
        f"--num_envs={40}", #256 default: 20, 60, 100, 200, 256 (the max I can do with rtx 4090), 400 (not working)
        f"--update_epochs={8}", 
        f"--num_minibatches={16}",
        f"--total_timesteps={10_000_000}",
        f"--eval_freq={10}",
        f"--num-steps={20}", # 20 by default
        f"--sim_quality={render_quality}",
        #f"--track"
    ])

    end_time = time.time()
    elapsed_sec = (end_time - start_time)
    print(f"Elasped (seconds): {elapsed_sec}")

    print("End")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual", action='store_true', help="Just state") # default is False

    args = parser.parse_args()

    if args.visual:
        tasks = [
            "PushCube-v1", 
            #"PickCube-v1", 
            #"StackCube-v1", 
            #"PegInsertionSide-v1", 
            #"AssemblingKits-v1", # realsense
            #"PlugCharger-v1"
        ]

        RENDER_QUALITY = "high" # rasterization
        MODALITY = "rgb"
        EXPERIMENT = f"octo/eval-success-rate" # modality or sim-quality etc.
        names = [
            f"{EXPERIMENT}-{MODALITY}-pushcube-{RENDER_QUALITY}", 
            #f"{EXPERIMENT}-{MODALITY}-pickcube-{RENDER_QUALITY}",
            #f"{EXPERIMENT}-{MODALITY}-stackcube-{RENDER_QUALITY}",
            #f"{EXPERIMENT}-{MODALITY}-peginsertionside-{RENDER_QUALITY}",
            #f"{EXPERIMENT}-{MODALITY}-assemblingkits-{RENDER_QUALITY}",
            #f"{EXPERIMENT}-{MODALITY}-plugcharger-{RENDER_QUALITY}", 
        ]

        assert len(tasks) == len(names), "equal number of params"

        print(f"Render quality is {RENDER_QUALITY}")

        for idx in range(len(tasks)):
            task_name = tasks[idx]
            name = names[idx]
            print(f"Running experiment for {task_name}-{name}")
            run_default_visual_octo(task=task_name, name=name, render_quality=RENDER_QUALITY)
            wait_time_s = 3
            print(f"Waiting for {wait_time_s} seconds ...")
            time.sleep(wait_time_s)
    else:
        print("No option has been selected")
