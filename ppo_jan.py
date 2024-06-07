import os
import random
import time
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


def layer_init_orthonomal(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def layer_init_uniform(layer, std=0, bias_const=0, init_w=3e-3):
    # std and bias_const get ignored
    torch.nn.init.uniform_(layer.weight, -init_w, init_w)
    torch.nn.init.constant_(layer.bias, 0)
    return layer


class Agent_Seperate_MLPs(nn.Module):
    def __init__(self, envs, init_type):
        if init_type == "orthogonal":
            layer_init = layer_init_orthonomal
        elif init_type == "uniform":
            layer_init = layer_init_uniform
        else:
            raise ValueError(
                f"Unknown init type {init_type}. Has to be either 'orthogonal' or 'uniform'"
            )

        super().__init__()
        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(
                nn.Linear(256, np.prod(envs.single_action_space.shape)),
                std=0.01 * np.sqrt(2),
            ),
        )
        self.actor_logstd = nn.Parameter(
            torch.ones(1, np.prod(envs.single_action_space.shape)) * -0.5
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action(self, x, deterministic=False):
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


class Agent_Shared_MLPs(nn.Module):
    def __init__(self, envs, init_type):
        if init_type == "orthogonal":
            layer_init = layer_init_orthonomal
        elif init_type == "uniform":
            layer_init = layer_init_uniform
        else:
            raise ValueError(
                f"Unknown init type {init_type}. Has to be either 'orthogonal' or 'uniform'"
            )

        super().__init__()

        self.network = nn.Sequential(
            layer_init(
                nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)
            ),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
        )

        self.actor_mean = layer_init(
            nn.Linear(256, np.prod(envs.single_action_space.shape)),
            std=0.01 * np.sqrt(2),
        )
        self.actor_logstd = nn.Parameter(
            torch.ones(1, np.prod(envs.single_action_space.shape)) * -0.5
        )
        self.critic = layer_init(nn.Linear(256, 1))

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action(self, x, deterministic=False):
        action_mean = self.actor_mean(self.network(x))
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(self.network(x))
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(self.network(x)),
        )


def run(
    num_envs: int = 512,
    num_steps: int = 50,
    total_timesteps: int = 10000000,
    num_minibatches: int = 32,
    evaluate: bool = False,
    track: bool = False,
    exp_name: Optional[str] = None,
    env_id: str = "PickCube-v1",
    seed: int = 1,
    torch_deterministic: bool = True,
    cuda: bool = True,
    wandb_project_name: Optional[str] = "jan_testing",
    wandb_entity: Optional[str] = None,
    capture_video: bool = True,
    save_train_video_freq: Optional[int] = None,
    num_eval_envs: int = 8,
    num_eval_steps: int = 50,
    checkpoint: Optional[str] = None,
    anneal_lr: bool = False,
    learning_rate: float = 3e-4,
    gamma: float = 0.8,
    gae_lambda: float = 0.9,
    finite_horizon_gae: bool = True,
    norm_adv: bool = True,
    clip_coef: float = 0.2,
    update_epochs: int = 4,
    target_kl: Optional[float] = 0.1,
    clip_vloss: bool = False,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    partial_reset: bool = True,
    eval_freq: int = 25,
    save_model: bool = True,
    special_rendering: bool = False,
    # OUR PARAMETERS TO TEST ONE
    action_transformation="tanh",
    layer_init="orthogonal",
    seperate_MLPs=True,
):
    batch_size = int(num_envs * num_steps)

    minibatch_size = int(batch_size // num_minibatches)
    num_iterations = total_timesteps // batch_size

    if exp_name is None:
        exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{env_id}__{exp_name}__{seed}__{int(time.time())}"
    else:
        run_name = exp_name

    writer = None
    if not evaluate:
        print("Running training")
        if track:
            import wandb

            wandb.init(
                project=wandb_project_name,
                entity=wandb_entity,
                sync_tensorboard=True,
                # config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s",
        )
    else:
        print("Running evaluation")

    # TRY NOT TO MODIFY: seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

    # env setup
    env_kwargs = dict(
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        sim_backend="gpu",
    )
    envs = gym.make(env_id, num_envs=num_envs if not evaluate else 1, **env_kwargs)
    if not special_rendering:
        eval_envs = gym.make(env_id, num_envs=num_eval_envs, **env_kwargs)
    else:
        eval_envs = gym.make(
            env_id, num_envs=num_eval_envs, shader_dir="rt", **env_kwargs
        )
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if evaluate:
            eval_output_dir = f"{os.path.dirname(checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        if save_train_video_freq is not None:
            save_video_trigger = lambda x: (x // num_steps) % save_train_video_freq == 0
            envs = RecordEpisode(
                envs,
                output_dir=f"runs/{run_name}/train_videos",
                save_trajectory=False,
                save_video_trigger=save_video_trigger,
                max_steps_per_video=num_steps,
                video_fps=30,
            )
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=eval_output_dir,
            save_trajectory=evaluate,
            trajectory_name="trajectory",
            max_steps_per_video=num_eval_steps,
            video_fps=30,
        )
    envs = ManiSkillVectorEnv(
        envs, num_envs, ignore_terminations=not partial_reset, **env_kwargs
    )
    eval_envs = ManiSkillVectorEnv(
        eval_envs, num_eval_envs, ignore_terminations=not partial_reset, **env_kwargs
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    if seperate_MLPs:
        agent = Agent_Seperate_MLPs(envs, layer_init).to(device)
    else:
        agent = Agent_Shared_MLPs(envs, layer_init).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(
        device
    )
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(
        device
    )
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=seed)
    eval_obs, _ = eval_envs.reset(seed=seed)
    next_done = torch.zeros(num_envs, device=device)
    eps_returns = torch.zeros(num_envs, dtype=torch.float, device=device)
    eps_lens = np.zeros(num_envs)
    place_rew = torch.zeros(num_envs, device=device)
    print(f"####")
    print(
        f"num_iterations={num_iterations} num_envs={num_envs} num_eval_envs={num_eval_envs}"
    )
    print(
        f"minibatch_size={minibatch_size} batch_size={batch_size} update_epochs={update_epochs}"
    )
    print(f"####")
    action_space_low, action_space_high = torch.from_numpy(
        envs.single_action_space.low
    ).to(device), torch.from_numpy(envs.single_action_space.high).to(device)

    def clip_action(action: torch.Tensor):
        return torch.clamp(action.detach(), action_space_low, action_space_high)

    def tanh_action(action: torch.Tensor):
        # don't pass gradients through tanh
        return torch.tanh(action).detach()

    # OUR CODE
    if action_transformation == "tanh":
        action_transformation = tanh_action
    elif action_transformation == "clip":
        action_transformation = clip_action
    else:
        raise ValueError(
            f"Unknown action transformation {action_transformation}. Has to be either 'tanh' or 'clip'"
        )

    if checkpoint:
        agent.load_state_dict(torch.load(checkpoint))

    for iteration in range(1, num_iterations + 1):
        print(f"Epoch: {iteration}, global_step={global_step}")
        final_values = torch.zeros((num_steps, num_envs), device=device)
        agent.eval()
        if iteration % eval_freq == 1:
            # evaluate
            print("Evaluating")
            eval_envs.reset()
            returns = []
            eps_lens = []
            successes = []
            failures = []
            # START OF EVALUATION
            for _ in range(num_eval_steps):
                with torch.no_grad():
                    (
                        eval_obs,
                        _,
                        eval_terminations,
                        eval_truncations,
                        eval_infos,
                    ) = eval_envs.step(agent.get_action(eval_obs, deterministic=True))
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        eps_lens.append(
                            eval_infos["final_info"]["elapsed_steps"][mask]
                            .cpu()
                            .numpy()
                        )
                        returns.append(
                            eval_infos["final_info"]["episode"]["r"][mask].cpu().numpy()
                        )
                        if "success" in eval_infos:
                            successes.append(
                                eval_infos["final_info"]["success"][mask].cpu().numpy()
                            )
                        if "fail" in eval_infos:
                            failures.append(
                                eval_infos["final_info"]["fail"][mask].cpu().numpy()
                            )
            returns = np.concatenate(returns)
            eps_lens = np.concatenate(eps_lens)
            print(
                f"Evaluated {num_eval_steps * num_envs} steps resulting in {len(eps_lens)} episodes"
            )
            if len(successes) > 0:
                successes = np.concatenate(successes)
                if writer is not None:
                    writer.add_scalar(
                        "charts/eval_success_rate", successes.mean(), global_step
                    )
                print(f"eval_success_rate={successes.mean()}")
            if len(failures) > 0:
                failures = np.concatenate(failures)
                if writer is not None:
                    writer.add_scalar(
                        "charts/eval_fail_rate", failures.mean(), global_step
                    )
                print(f"eval_fail_rate={failures.mean()}")

            print(f"eval_episodic_return={returns.mean()}")
            if writer is not None:
                writer.add_scalar(
                    "charts/eval_episodic_return_per_step", returns.mean(), global_step
                )
                writer.add_scalar(
                    "charts/eval_episodic_return_per_episode", returns.mean(), iteration
                )
                writer.add_scalar(
                    "charts/eval_episodic_length", eps_lens.mean(), global_step
                )
            if evaluate:
                break

        # END OF EVALUATION
        if save_model and iteration % eval_freq == 1:
            model_path = f"runs/{run_name}/ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        # Annealing the rate if instructed to do so.
        if anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # ROLLOUT TO GATHER DATA
        rollout_time = time.time()
        for step in range(0, num_steps):
            global_step += num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action_transformation(action)
            )
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1)

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                episodic_return = (
                    final_info["episode"]["r"][done_mask].cpu().numpy().mean()
                )
                if "success" in final_info:
                    writer.add_scalar(
                        "charts/success_rate",
                        final_info["success"][done_mask].cpu().numpy().mean(),
                        global_step,
                    )
                if "fail" in final_info:
                    writer.add_scalar(
                        "charts/fail_rate",
                        final_info["fail"][done_mask].cpu().numpy().mean(),
                        global_step,
                    )
                writer.add_scalar(
                    "charts/episodic_return_per_step", episodic_return, global_step
                )
                writer.add_scalar(
                    "charts/episodic_return_per_episode", episodic_return, iteration
                )
                writer.add_scalar(
                    "charts/episodic_length",
                    final_info["elapsed_steps"][done_mask].cpu().numpy().mean(),
                    global_step,
                )

                final_values[
                    step, torch.arange(num_envs, device=device)[done_mask]
                ] = agent.get_value(infos["final_observation"][done_mask]).view(-1)
        rollout_time = time.time() - rollout_time

        # bootstrap value according to termination and truncation
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t]

                if finite_horizon_gae:
                    if t == num_steps - 1:
                        lam_coef_sum = 0.0
                        reward_term_sum = 0.0
                        value_term_sum = 0.0
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + gae_lambda * lam_coef_sum
                    reward_term_sum = (
                        gae_lambda * gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    )
                    value_term_sum = (
                        gae_lambda * gamma * value_term_sum + gamma * real_next_values
                    )

                    advantages[t] = (
                        reward_term_sum + value_term_sum
                    ) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = (
                        delta + gamma * gae_lambda * next_not_done * lastgaelam
                    )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        agent.train()
        b_inds = np.arange(batch_size)
        clipfracs = []
        update_time = time.time()
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                    ]

                if target_kl is not None and approx_kl > target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:  # doesnt run by default
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

            if target_kl is not None and approx_kl > target_kl:
                break
        update_time = time.time() - update_time

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )
        writer.add_scalar("charts/update_time", update_time, global_step)
        writer.add_scalar("charts/rollout_time", rollout_time, global_step)
        writer.add_scalar(
            "charts/rollout_fps", num_envs * num_steps / rollout_time, global_step
        )

    if not evaluate:
        if save_model:
            model_path = f"runs/{run_name}/final_ckpt.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        writer.close()
    envs.close()
    eval_envs.close()
