# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time

import io
from PIL import Image

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

import trimesh
import trimesh.scene

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# Custom utils
from agents import compute_GAE, FlattenPointcloudObservationWrapper
from agents import PointcloudAgent
from data_utils import DictArray
from sim_utils import set_simulation_quality
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

    # Varying simulation/rendering params (experiment)
    RENDER_TYPE = args.sim_quality
    ENABLE_SHADOWS = set_simulation_quality(RENDER_TYPE)

    from custom_tasks import *
    print("Randomize existing camera poses")
    tasks_mapping = {
        "PullCube-v1": "PullCube-pcd",
        "PushCube-v1": "PushCube-pcd",
        "PickCube-v1": "PickCube-pcd",
        "StackCube-v1": "StackCube-pcd",
        "PegInsertionSide-v1": "PegInsertionSide-pcd",
        "AssemblingKits-v1": "AssemblingKits-pcd",
        "PlugCharger-v1": "PlugCharger-pcd"
    }

    args.env_id = tasks_mapping[args.env_id]
    args.exp_name = args.exp_name + "-pcd"
    


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

    # env setup
    env_kwargs = dict(
        obs_mode="pointcloud", 
        control_mode="pd_joint_delta_pos", 
        render_mode="rgb_array", 
        sim_backend="gpu",
        enable_shadow=ENABLE_SHADOWS,
    )
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, **env_kwargs)
    envs = gym.make(
        args.env_id, 
        num_envs=args.num_envs if not args.evaluate else 1, 
        **env_kwargs
    )

    # rgbd obs mode returns a dict of data, we flatten it so there is just a rgbd key and state key
    envs = FlattenPointcloudObservationWrapper(envs, pointcloud_only=True)
    eval_envs = FlattenPointcloudObservationWrapper(eval_envs, pointcloud_only=True)

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
    
    # Maniskill wrapper around gym
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=False, **env_kwargs)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=False, **env_kwargs)
    
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # NOTE: ALGO Logic: Storage setup
    obs = DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # NOTE: TRY NOT TO MODIFY: start the game
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
    
    # NOTE: Pointcloud encoder (setup agent/s)
    agent = PointcloudAgent(envs, sample_obs=next_obs, is_tracked=args.track).agent.to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))



    # NOTE: number of iterations is total_timesteps / batch_size
    for iteration in range(1, args.num_iterations + 1):
        print(f"Epoch: {iteration}, global_step={global_step}") #---------------------------------- time start

        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)

        # EVALUATION ------------------------------------------------------------------------------
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
                    eval_obs, _, eval_terminations, eval_truncations, eval_infos = \
                        eval_envs.step(agent.get_action(eval_obs, deterministic=True))
                    
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

            print(f"eval_episodic_return (reward)={returns.mean()}")
            
            if writer is not None:
                writer.add_scalar("charts/eval_episodic_return (reward)", returns.mean(), global_step)
                writer.add_scalar("charts/eval_episodic_length (time per episode)", eps_lens.mean(), global_step)
            
            if args.evaluate:
                break
        # EVALUATION ends -------------------------------------------------------------------------------------------




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




        # NOTE: Main loop (train) --------------------------------------------------------------------------------------
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done



            # NOTE: Logging (pointcloud to mesh)
            xyz_log = obs["pcd"][0, ..., :3].detach().cpu().numpy()[0]
            colors_log = obs["pcd"][0, ..., 3:].detach().cpu().numpy() if obs["pcd"].shape[-1] > 3 else \
                np.zeros_like(xyz_log)[0]

            # debug
            import pickle
            pcd_log = {
                "xyz": xyz_log,
                "rgb": colors_log,
            }
            with open('pcd_log_TEST.pickle', 'wb') as handle:
                pickle.dump(pcd_log, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # debug

            pcd = trimesh.points.PointCloud(xyz_log, colors_log)
            UID = 0
            fov_angle = np.rad2deg(np.pi / 2)
            mesh_camera = trimesh.scene.Camera(UID, (1024, 1024), fov=(fov_angle, fov_angle))
            cam2world = np.array(
                [[ 0.        , -0.78086877,  0.62469506,  0.3       ],  
                [ 1.        ,  0.        ,  0.        ,  0.        ],
                [ 0.        ,  0.62469506,  0.7808688 ,  0.6       ],
                [ 0.        ,  0.        ,  0.        ,  1.        ]]
            )
            mesh_scene = trimesh.Scene([pcd], camera=mesh_camera, camera_transform=cam2world)
            rendered_img = mesh_scene.save_image(resolution=(800, 600))
            img_log = np.array(rendered_img)
            
            if img_log.shape[-1] > 3:
                img_log = img_log[..., :3]
            wandb.log({
                f"obs[{step}]": wandb.Image(img_log)
            })
            #writer.add_image(f"observations_{step}", tf_rgb_log)




            # ALGO LOGIC: action logic

            # (1) Get action and value for given observation
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            # (2) Given an action step forward
            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1)





            # NOTE: Logging
            gpu_allocated_mem = memory_logger.get_gpu_allocated_memory()
            cpu_allocated_mem = memory_logger.get_cpu_allocated_memory()
            wandb.log({
                "gpu_alloc_mem": gpu_allocated_mem,
                "cpu_alloc_mem": cpu_allocated_mem,
            })




            # (3) Check if we are done
            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                episodic_return = final_info['episode']['r'][done_mask].mean().cpu().numpy()

                writer.add_scalar("charts/success_rate", final_info["success"][done_mask].float().mean().cpu().numpy(), global_step)
                writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                writer.add_scalar("charts/episodic_length", final_info["elapsed_steps"][done_mask].float().mean().cpu().numpy(), global_step)

                for k in infos["final_observation"]:
                    infos["final_observation"][k] = infos["final_observation"][k][done_mask]
                
                final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(infos["final_observation"]).view(-1)
        
        rollout_time = time.time() - rollout_time
        





        # bootstrap value according to termination and truncation
        # (4) Given the current action/observation, compute advatnage to update policies
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0

            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else: # not the last row in rewards
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                
                real_next_values = next_not_done * nextvalues + final_values[t] # t instead of t+1
                # next_not_done means nextvalues is computed from the correct next_obs
                # if next_not_done is 1, final_values is always 0
                # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                if args.finite_horizon_gae:
                    advantages[t] = compute_GAE(
                        t, args.num_steps, 
                        next_not_done, 
                        args.gae_lambda, args.gamma, 
                        rewards,
                        real_next_values, 
                        values
                    )
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
        # (5) Train the policies based on advantage of given x|a
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

                # POLICY loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # VALUE loss
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




        # NOTE: Logging
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