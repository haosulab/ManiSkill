import os
import random
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro

# ManiSkill specific imports
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# Custom utils
from agents import Agent
from custom_tasks import *
from sim_utils import *
from visual_args import Args

# NOTE: This is experimental code that if successful is merged into ppo_rgb_custom.py


if __name__ == "__main__":
    args = tyro.cli(Args)

    args.env_id = "PushCube-v1"
    args.exp_name = "experimental-pushcube"
    
    args.num_envs = 1
    args.num_steps = 5
    args.num_iterations = 100




    # EXPERIMENTS (with sim params)
    # TODO: Varying simulation/rendering params (experiment)
    # (1) Simulation quality
    args.sim_quality = "high"
    ENABLE_SHADOWS = set_simulation_quality(args.sim_quality)

    print("Randomize existing camera poses")
    tasks_mapping = {
        "PushCube-v1": "PushCube-Randomization",
    }
    if args.env_id not in tasks_mapping:
        raise(f"{args.env_id} not in supported tasks")

    args.env_id = tasks_mapping[args.env_id]
    args.exp_name = args.exp_name + "-randomization"
    

    # NOTE: Happens at start
    # (2) Camera resolution
    resolution = SimulationQuantities.RESOLUTIONS[2]
    print(f"Chosen resolution: {resolution}")
    # numpy convention -> (h, w)
    sensor_configs = dict(width=resolution[1], height=resolution[0])

    # (3) Light properties
    light_color = SimulationQuantities.LIGHT_COLORS[0]
    light_directions = [SimulationQuantities.LIGHT_DIRECTIONS[0]]

    # (4) Material properties
    specularity = SimulationQuantities.SPECULARITY[-1]
    metallicity = SimulationQuantities.METALLICITY[0]
    index_of_refraction = SimulationQuantities.INDEX_OF_REFRACTION[1]
    transmission = SimulationQuantities.TRANSMISSION[0]
    material_color = SimulationQuantities.MATERIAL_COLORS[0]

    # (5) Material color
    material_color = SimulationQuantities.MATERIAL_COLORS[0]

    # (6) Material physics properties
    mass = None
    density = None

    CHANGE_TARGET = True

    sim_params = dict(
        sensor_configs=sensor_configs,
        mass=mass,
        density=density,
        specularity=specularity,
        metallicity=metallicity,
        ior=index_of_refraction,
        transmission=transmission,
        material_color=material_color,
        light_color=light_color,
        light_directions = light_directions,
        change_target=CHANGE_TARGET
    )

    env_kwargs = dict(
        obs_mode="rgbd", 
        control_mode="pd_joint_delta_pos", 
        render_mode="rgb_array", 
        sim_backend="gpu",
        enable_shadow=ENABLE_SHADOWS,
        sim_params=sim_params
    )





    # ------------------------------------------------------------------------------------------------------------------
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ENVIRONMENT SETUP
    print("Creating environment\n")
    envs = gym.make(
        args.env_id, 
        num_envs=args.num_envs, 
        **env_kwargs
    )

    # rgbd obs mode returns a dict of data, we flatten it so there is just a rgbd key and state key
    WITH_STATE = False # NOTE: rgb + state or rgb
    envs = FlattenRGBDObservationWrapper(envs, rgb_only=True)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)

    if args.capture_video:
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(
                envs, 
                output_dir=f"runs/{run_name}/experiment_videos", 
                save_trajectory=False, 
                save_video_trigger=save_video_trigger, 
                max_steps_per_video=args.num_steps, 
                video_fps=30
            )
    
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=False, **env_kwargs)
    
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"


    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)

    print(f"####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"####")

    ENABLE_WANDB = False
    SAVE_IMGS = False

    # Wandb
    if ENABLE_WANDB:
        import wandb
        wandb.login()
        wandb.init(
            project="Raycast vs Rasterization ENABLE_WANDB",
            entity="embarc_lab",
            sync_tensorboard=False,
        )

        # Memory logging
        import sys
        sys.path.append("./")
        sys.path.append("../")
        sys.path.append("../scripts/")
        sys.path.append("scripts/")
        from mem_logger import MemLogger
        LOG_PATH = "/home/filip-grigorov/code/scripts/data/"
        memory_logger = MemLogger(
            device="cuda" if torch.cuda.is_available() else "cpu", 
            dump_after=100, 
            gpu_mem_logs_path=LOG_PATH, 
            cpu_mem_logs_path=LOG_PATH, 
            loss_logs_path=LOG_PATH, 
            lbl="train_visual"
        )

    time_pnts = []

    for iteration in range(1, args.num_iterations + 1):
        frames = []

        print(f"Epoch: {iteration}, global_step={global_step}")
        rollout_time = time.time()

        for step in range(0, args.num_steps):
            with torch.no_grad():
                probs = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
                action = probs.sample(sample_shape=[1, 8])
                #print(f"Sampled action: {action}")

            step_time_pnt = time.time()

            next_obs, reward, terminations, truncations, infos = envs.step(action)

            elapsed_time_ms = (time.time() - step_time_pnt) * 1e-3

            time_pnts.append(elapsed_time_ms)

            if ENABLE_WANDB:
                wandb.log({
                    "ENABLE_WANDB/elapsed_time_pert_step() (ms)": elapsed_time_ms,
                })

            # Logging
            if ENABLE_WANDB:
                gpu_allocated_mem = memory_logger.get_gpu_allocated_memory()
                cpu_allocated_mem = memory_logger.get_cpu_allocated_memory()
                wandb.log({
                    "ENABLE_WANDB/gpu_alloc_mem": gpu_allocated_mem,
                    "ENABLE_WANDB/cpu_alloc_mem": cpu_allocated_mem,
                })

            if SAVE_IMGS:
                img_log = next_obs["rgb"].detach()[0].cpu().numpy()
                if img_log.shape[-1] > 3:
                    img_log = img_log[..., :3]
                frames.append(img_log)

        rollout_time = time.time() - rollout_time
        if ENABLE_WANDB:
            wandb.log({
                "ENABLE_WANDB/rollout_time (s)": (time.time() - step_time_pnt),
            })
        print(f"Rollout complete in {rollout_time} secs!")

        # Log data here
        if SAVE_IMGS:
            fig, axes = plt.subplots(1, args.num_steps, figsize=(50, 50))
            for idx in range(args.num_steps):
                axes[idx].imshow(frames[idx])
            plt.savefig(f"visualizations/env_sim_params_imgs_{iteration + 1}.png", dpi=300, bbox_inches='tight')

    print("------------ Summary ------------")
    mean = np.mean(time_pnts)
    std = np.std(time_pnts)
    min = np.min(time_pnts)
    max = np.max(time_pnts)
    print(f"Elapsed time (ms): \nmean: {mean}\nstd: {std}\nmin: {min}\nmax: {max}\n\n\n")

    envs.close()
