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
from sim_utils import set_simulation_quality
from visual_args import Args

# NOTE: This is experimental code that if successful is merged into ppo_rgb_custom.py


if __name__ == "__main__":
    args = tyro.cli(Args)

    args.env_id = "PushCube-v1"
    args.exp_name = "experimental-pushcube"
    
    args.num_envs = 1
    args.num_steps = 5
    args.num_iterations = 1




    # EXPERIMENTS (with sim params)
    # TODO: Varying simulation/rendering params (experiment)
    # (1) Simulation quality
    RENDER_TYPE = args.sim_quality
    ENABLE_SHADOWS = set_simulation_quality(RENDER_TYPE)

    print("Randomize existing camera poses")
    tasks_mapping = {
        "PushCube-v1": "PushCube-Randomization",
    }
    if args.env_id not in tasks_mapping:
        raise(f"{args.env_id} not in supported tasks")

    args.env_id = tasks_mapping[args.env_id]
    args.exp_name = args.exp_name + "-randomization"
    

    # NOTE: Happens at start
    # (2) Randomize camera resolution
    RANDOMIZE_RESOLUTION = False
    resolutions = [(128, 128), (256, 256), (512, 512), (1024, 1024)]
    random_resolution_pick = int(np.random.uniform(low=0, high=len(resolutions) - 1)) if RANDOMIZE_RESOLUTION else -2
    random_resolution_pair = resolutions[random_resolution_pick]
    print(f"Chosen resolution: {random_resolution_pair}")
    # numpy convention -> (h, w)
    sensor_configs = dict(width=random_resolution_pair[1], height=random_resolution_pair[0])

    # (3) Randomize light properties
    #TODO

    # (4) Randomize material properties
    RANDOMIZE_MATERIAL = False
    specularity = [0.0, 0.5, 1.0] # 0.0 is mirror-like
    metallicity = [0.0, 1.0] # 0.0 is for non-metals and 1.0 is for metals
    index_of_refraction = [1.0, 1.4500000476837158, 1.9]
    transmission = [0.0, 0.5, 1.0]
    material_colors = [
        # R,G,B,A
        np.array([12, 42, 160, 255]), 
        np.array([160, 42, 12, 255]), 
        np.array([12, 160, 42, 255]), 
        np.array([12, 42, 160, 100]), 
    ]

    random_idx = int(np.random.uniform(low=0, high=len(specularity) - 1)) if RANDOMIZE_MATERIAL else -1
    random_specular_val = specularity[random_idx]
    random_metallic_val = metallicity[random_idx]
    random_ior = index_of_refraction[random_idx]
    random_transmission = transmission[random_idx]

    RANDOMIZE_MATERIAL_COLOR = False
    random_idx = int(np.random.uniform(low=0, high=len(material_colors) - 1)) if RANDOMIZE_MATERIAL_COLOR else -1
    material_color = material_colors[random_idx]

    sim_params = dict(
        sensor_configs=sensor_configs,
        randomize_lights=False,
        randomize_physics=False,
        randomize_material=True,
        mass=None,
        specularity=random_specular_val,
        metallicity=random_metallic_val,
        ior=random_ior,
        transmission=random_transmission,
        material_color=material_color,
        light_color=[]
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
    WITH_STATE = True # NOTE: rgb + state or rgb
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


    for iteration in range(1, args.num_iterations + 1):
        frames = []

        print(f"Epoch: {iteration}, global_step={global_step}")
        rollout_time = time.time()

        for step in range(0, args.num_steps):
            with torch.no_grad():
                probs = torch.distributions.normal.Normal(loc=0.0, scale=1.0)
                action = probs.sample(sample_shape=[1, 8])
                print(f"Sampled action: {action}")
            next_obs, reward, terminations, truncations, infos = envs.step(action)

            img_log = next_obs["rgb"].detach()[0].cpu().numpy()
            if img_log.shape[-1] > 3:
                img_log = img_log[..., :3]
            frames.append(img_log)

        rollout_time = time.time() - rollout_time
        print(f"Rollout complete in {rollout_time} secs!")

        # Log data here
        fig, axes = plt.subplots(1, args.num_steps, figsize=(50, 50))
        for idx in range(args.num_steps):
            axes[idx].imshow(frames[idx])
        plt.savefig(f"visualizations/env_sim_params_imgs_{iteration + 1}.png", dpi=300, bbox_inches='tight')

    envs.close()
