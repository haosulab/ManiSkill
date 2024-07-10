    # docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# Custom utils
from sim_utils import *
from visual_args import Args



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

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# export DISPLAY=:0.0
os.environ['DISPLAY'] = ':0.0'

from octo.model.octo_model import OctoModel



if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    print(f"Batch size: {args.batch_size}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Number of iters: {args.num_iterations}")

    # Varying simulation/rendering params (experiment)
    RENDER_TYPE = args.sim_quality
    ENABLE_SHADOWS = set_simulation_quality(RENDER_TYPE)

    # Camera resolution
    CAM_IDX = -1
    RESOLUTION = SimulationQuantities.RESOLUTIONS[CAM_IDX]
    sensor_configs = dict(width=RESOLUTION[0], height=RESOLUTION[1])
    print(f"Camera resolution: {RESOLUTION}")

    # Possible randomizations
    sim_params = {}
    if args.random_cam_pose:
        from custom_tasks import *
        print("Randomize existing camera poses")
        tasks_mapping = {
            "PullCube-v1": "PullCube-RandomCameraPose",
            "PushCube-v1": "PushCube-RandomCameraPose",
            "PickCube-v1": "PickCube-RandomCameraPose",
            "StackCube-v1": "StackCube-RandomCameraPose",
            "PegInsertionSide-v1": "PegInsertionSide-RandomCameraPose",
            "AssemblingKits-v1": "AssemblingKits-RandomCameraPose",
            "PlugCharger-v1": "PlugCharger-RandomCameraPose"
        }

        args.env_id = tasks_mapping[args.env_id]
        args.exp_name = args.exp_name + "-random-cam-pose"

    elif args.vary_sim_parameters:
        from custom_tasks import *
        print("Randomize existing camera poses")
        tasks_mapping = {
            "PullCube-v1": "PullCube-Randomization",
            "PushCube-v1": "PushCube-Randomization",
            "PickCube-v1": "PickCube-Randomization",
            "StackCube-v1": "StackCube-Randomization",
            "PegInsertionSide-v1": "PegInsertionSide-Randomization",
            "AssemblingKits-v1": "AssemblingKits-Randomization",
            "PlugCharger-v1": "PlugCharger-Randomization"
        }

        args.env_id = tasks_mapping[args.env_id]
        args.exp_name = args.exp_name + "-randomization"

        # (1) Light properties
        light_color = SimulationQuantities.LIGHT_COLORS[0]
        light_directions = [SimulationQuantities.LIGHT_DIRECTIONS[0]]

        # (2) Material properties
        specularity = SimulationQuantities.SPECULARITY[-1]
        metallicity = SimulationQuantities.METALLICITY[0]
        index_of_refraction = SimulationQuantities.INDEX_OF_REFRACTION[1]
        transmission = SimulationQuantities.TRANSMISSION[0]
        material_color = SimulationQuantities.MATERIAL_COLORS[0]

        # (3) Material color
        material_color = SimulationQuantities.MATERIAL_COLORS[0]

        # (4) Material physics properties
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


    


    # ------------------------------------------------------------------------------------------------------------------
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    writer = None

    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            wandb.login()
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )

        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    else:
        print("Running evaluation")


    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Environment setup !!!
    env_kwargs = dict(
        obs_mode="rgbd", 
        control_mode="pd_joint_delta_pos", 
        render_mode="rgb_array", 
        sim_backend="gpu",
        enable_shadow=ENABLE_SHADOWS,
        sensor_configs=sensor_configs
    )

    # NOTE: Possible randomization
    if args.vary_sim_parameters:
        print("Setting up custom simulation parameters")
        env_kwargs = dict(
            obs_mode="rgbd", 
            control_mode="pd_joint_delta_pos", 
            render_mode="rgb_array", 
            sim_backend="gpu",
            enable_shadow=ENABLE_SHADOWS,
            sensor_configs=sensor_configs,
            sim_params=sim_params
        )

    # Eval
    eval_envs = gym.make(
        args.env_id, 
        num_envs=args.num_eval_envs, 
        **env_kwargs
    )

    # rgbd obs mode returns a dict of data, we flatten it so there is just a rgbd key and state key
    WITH_STATE = False # NOTE: rgb + state or rgb
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb_only=True)

    if isinstance(eval_envs.action_space, gym.spaces.Dict):
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.evaluate, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
    
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=False, **env_kwargs)
    
    assert isinstance(eval_envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()

    print(f"####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    print(f"####")

    model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")
    print(model.get_pretty_spec())

    for iteration in range(1, args.num_iterations + 1):
        print(f"Epoch: {iteration}, global_step={global_step}")

        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        if iteration % args.eval_freq == 1:
            # evaluate
            print("Evaluating")
            eval_envs.reset()
            returns = []
            eps_lens = []
            successes = []
            failures = []

            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    raw_action, action = model.step(eval_obs)
                    eval_obs, _, eval_terminations, eval_truncations, eval_infos = \
                        eval_envs.step(
                            np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]])
                        )
                    
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        eps_lens.append(eval_infos["final_info"]["elapsed_steps"][mask].cpu().numpy())
                        returns.append(eval_infos["final_info"]["episode"]["r"][mask].cpu().numpy())

                        if "success" in eval_infos:
                            successes.append(eval_infos["final_info"]["success"][mask].cpu().numpy())
                        
                        if "fail" in eval_infos:
                            failures.append(eval_infos["final_info"]["fail"][mask].cpu().numpy())
            
            returns = np.concatenate(returns) if len(returns) > 0 else np.array(returns)
            eps_lens = np.concatenate(eps_lens) if len(eps_lens) > 0 else np.array(eps_lens)
            print(f"Evaluated {args.num_eval_steps * args.num_envs} steps resulting in {len(eps_lens)} episodes")
            
            if len(successes) > 0:
                successes = np.concatenate(successes)
                if writer is not None: writer.add_scalar("charts/eval_success_rate", successes.mean(), global_step)
                print(f"eval_success_rate={successes.mean()}")
            
            if len(failures) > 0:
                failures = np.concatenate(failures)
                if writer is not None: writer.add_scalar("charts/eval_fail_rate", failures.mean(), global_step)
                print(f"eval_fail_rate={failures.mean()}")

            print(f"eval_episodic_return={returns.mean()}")
            
            if writer is not None:
                writer.add_scalar("charts/eval_episodic_return", returns.mean(), global_step)
                writer.add_scalar("charts/eval_episodic_length", eps_lens.mean(), global_step)
            
            if args.evaluate:
                break

    envs.close()

    if writer is not None: writer.close()

    wandb.finish()

if __name__ == "__main__":
    run()
