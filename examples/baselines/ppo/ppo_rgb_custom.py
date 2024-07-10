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
from agents import Agent
from data_utils import DictArray
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
    CAM_IDX = 0 # -1
    RESOLUTION = SimulationQuantities.RESOLUTIONS[CAM_IDX]
    sensor_configs = dict(width=RESOLUTION[0], height=RESOLUTION[1])
    print(f"Camera resolution: {RESOLUTION}")

    # Possible randomizations
    sim_params = {}
    if args.random_cam_pose:
        from custom_tasks import *
        print("random_cam_pose")
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
        print("vary_sim_parameters")
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
    
    # Train
    envs = gym.make(
        args.env_id, 
        num_envs=args.num_envs if not args.evaluate else 1, 
        **env_kwargs
    )

    # rgbd obs mode returns a dict of data, we flatten it so there is just a rgbd key and state key
    PRETRAINED = True
    WITH_STATE = False # NOTE: rgb + state or rgb
    envs = FlattenRGBDObservationWrapper(envs, rgb_only=False)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb_only=False)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.evaluate, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
    
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=False, **env_kwargs)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=False, **env_kwargs)
    
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # ALGO Logic: Storage setup
    obs = DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    eps_returns = torch.zeros(args.num_envs, dtype=torch.float, device=device)
    eps_lens = np.zeros(args.num_envs)
    place_rew = torch.zeros(args.num_envs, device=device)

    print(f"####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    print(f"####")
    
    agent = Agent(envs, sample_obs=next_obs, is_tracked=args.track, with_state=WITH_STATE, pretrained=PRETRAINED).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        agent.load_state_dict(torch.load(args.checkpoint))

    for iteration in range(1, args.num_iterations + 1):
        print(f"Epoch: {iteration}, global_step={global_step}")

        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()
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
                    eval_obs, _, eval_terminations, eval_truncations, eval_infos = eval_envs.step(agent.get_action(eval_obs, deterministic=True))
                    
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




        if args.save_model and iteration % args.eval_freq == 1:
            model_path = f"runs/{run_name}/ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        rollout_time = time.time()

        # NOTE: Main loop
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # NOTE: Logging
            if args.track:
                tf_rgb_log = obs[step]["rgbd"].detach()[0].cpu().numpy()
                if tf_rgb_log.shape[-1] > 3:
                    tf_rgb_log = tf_rgb_log[..., :3]
                wandb.log({
                    f"obs[{step}]": wandb.Image(tf_rgb_log)
                })
            #writer.add_image(f"observations_{step}", tf_rgb_log)

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action)
            # plot the chosen actions from the normal distribution
            # experiment
            writer.add_scalar("sample/mean_chosen_action", action.mean(), global_step)

            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1)

            # NOTE: Logging
            if args.track:
                gpu_allocated_mem = memory_logger.get_gpu_allocated_memory()
                cpu_allocated_mem = memory_logger.get_cpu_allocated_memory()
                wandb.log({
                    "gpu_alloc_mem": gpu_allocated_mem,
                    "cpu_alloc_mem": cpu_allocated_mem,
                })

            # debug
            if (infos['is_obj_placed'].any() and infos['is_robot_static'].any()) and infos['is_grasped'].any():
                print(infos.keys())
                if "final_info" in infos:
                    print(f"\n\t\t\t reward: {infos['final_info']['episode']['r'][infos['_final_info']].mean().cpu().numpy()}")
                    print(f"\n\t\t\t success rate: {infos['final_info']['success'][infos['_final_info']].float().mean().cpu().numpy()}")

                print(f"\n\t\t\t is_obj_placed: {infos['is_obj_placed']}")
                print(f"\n\t\t\t is_robot_static: {infos['is_robot_static']}")
                print(f"\n\t\t\t success: {infos['success']}")
                print(f"\n\t\t\t is_grasped: {infos['is_grasped']}\n\n\n")

                raise("debug")
            # debug

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                episodic_return = final_info['episode']['r'][done_mask].mean().cpu().numpy()

                writer.add_scalar("charts/success_rate", final_info["success"][done_mask].float().mean().cpu().numpy(), global_step)
                writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                writer.add_scalar("charts/episodic_length", final_info["elapsed_steps"][done_mask].float().mean().cpu().numpy(), global_step)

                for k in infos["final_observation"]:
                    infos["final_observation"][k] = infos["final_observation"][k][done_mask]
                
                num_envs_range_for_done = torch.arange(args.num_envs, device=device)[done_mask]
                final_values[step, num_envs_range_for_done] = agent.get_value(infos["final_observation"]).view(-1)
        
        rollout_time = time.time() - rollout_time
        
        # bootstrap value according to termination and truncation
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0

            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                
                real_next_values = next_not_done * nextvalues + final_values[t] # t instead of t+1
                # next_not_done means nextvalues is computed from the correct next_obs
                # if next_not_done is 1, final_values is always 0
                # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                if args.finite_horizon_gae:
                    """
                    See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                    1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                    lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                    lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                    lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                    We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                    """
                    if t == args.num_steps - 1: # initialize
                        lam_coef_sum = 0.
                        reward_term_sum = 0. # the sum of the second term
                        value_term_sum = 0. # the sum of the third term
                    
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                     # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam
            
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # NOTE: Optimizing the policy and value network
        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        update_time = time.time()

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        update_time = time.time() - update_time
        
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        print("SPS:", int(global_step / (time.time() - start_time)))
        
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar("charts/update_time", update_time, global_step)
        writer.add_scalar("charts/rollout_time", rollout_time, global_step)
        writer.add_scalar("charts/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)

    if args.save_model and not args.evaluate:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()

    if writer is not None: writer.close()

    wandb.finish()