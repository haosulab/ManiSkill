# Import required packages
import os
import argparse
import random
import time
import gymnasium as gym
from gymnasium import spaces, Wrapper
import numpy as np
import mani_skill2.envs
import matplotlib.pyplot as plt
from distutils.util import strtobool

os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.normal import Normal

# import datetime
from collections import defaultdict
# from functools import partial

import mani_skill2.envs
from mani_skill2.utils.common import flatten_state_dict, flatten_dict_space_keys
# from mani_skill2.utils.wrappers import RecordEpisode
from mani_skill2.vector.vec_env import VecEnvObservationWrapper
from mani_skill2.vector.VisualEncoder import VisualEncoder
# from dataclasses import dataclass
from mani_skill2.vector.wrappers.sb3 import select_index_from_dict



ALGO_NAME = "PPO"

class AutoResetVecEnvWrapper(Wrapper):
    # adapted from https://github.com/haosulab/ManiSkill2/blob/main/mani_skill2/vector/wrappers/sb3.py#L25
    def step(self, actions):
        vec_obs, rews, dones, truncations, infos = self.env.step(actions)
        if (not dones.any()) and (not truncations.any()):
            return vec_obs, rews, dones, truncations, infos

        for i, truncated_ in enumerate(truncations):
            if truncated_:
                # NOTE: ensure that it will not be inplace modified when reset
                infos[i]["terminal_observation"] = select_index_from_dict(vec_obs, i)

        for i, done_ in enumerate(dones):
            if done_:
                # NOTE: ensure that it will not be inplace modified when reset
                infos[i]["terminal_observation"] = select_index_from_dict(vec_obs, i)

        reset_indices = np.where(np.logical_or(dones, truncations))[0]
        vec_obs, _ = self.env.reset(indices=reset_indices)
        return vec_obs, rews, dones, truncations, infos

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="VisualEncoderExample",
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ManiSkill2-dev",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="PickCube-v3",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=20_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=16,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=250, # this hp is pretty important
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.8, # important
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.9,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=16,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=20,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=0.2,
        help="the target KL divergence threshold")
    
    # Maniskill
    parser.add_argument("--control-mode", type=str, default='pd_ee_delta_pos')

    # special to VisualEncoder
    parser.add_argument("--observation-mode", type=str, default='both',
        help="After VisualEncoder, obs will be a dict containing keys 'state' and 'image'. Use 'both' to include both as input to process_obs_dict().")

    args = parser.parse_args()
    args.algo_name = ALGO_NAME
    args.script = __file__
    args.batch_size = int(args.num_envs * args.num_steps) # the batch size
    args.minibatch_size = int(args.batch_size // args.num_minibatches) # the mini-batch size
    args.num_iterations = args.total_timesteps // args.batch_size # the number of iterations
    return args


def flatten_space_dict_keys(d: dict, prefix=""):
    """Flatten a space dict by expanding its keys recursively."""
    out = dict()
    for k, v in d.items():
        if isinstance(v, spaces.dict.Dict):
            out.update(flatten_space_dict_keys(v, prefix + k + "/"))
        else:
            out[prefix + k] = v
    return out

def seed_env(env, seed):
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)

class Agent(nn.Module):
    def __init__(self, envs, obs_shape):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.prod(obs_shape), 1024)),
            nn.Tanh(),
            layer_init(nn.Linear(1024, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(envs.embedding_size, 1024)),
            nn.Tanh(),
            layer_init(nn.Linear(1024, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    
def collect_episode_info(info, result=None):
    if result is None:
        result = defaultdict(list)
    for item in info:
        if "episode" in item.keys():
            print(f"global_step={global_step}, episodic_return={item['episode']['r']}, success={item['success']}")
            result['return'].append(item['episode']['r'])
            result['len'].append(item["episode"]["l"])
            result['success'].append(item['success'])
    return result

# process the vector env operation based on args.observation_mode. This is called after receiving the obs dict from vector env.
def process_obs_dict(obs_dict, observation_mode):
    if observation_mode == "state":
        return obs_dict["state"]
    elif observation_mode == "image":
        return obs_dict["embedding"]
    elif observation_mode == "both":
        return torch.cat([obs_dict["state"], obs_dict["embedding"]], dim=-1)

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb

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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    wrappers = [gym.wrappers.RecordEpisodeStatistics, gym.wrappers.ClipAction,]
    envs = mani_skill2.vector.make(
            args.env_id, args.num_envs, obs_mode='rgbd', reward_mode='dense', control_mode=args.control_mode, wrappers=wrappers, # camera_cfgs=cam_cfg,
    )
    envs.is_vector_env = True
    envs = VisualEncoder(envs, encoder='r3m')
    envs = AutoResetVecEnvWrapper(envs)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    print("Envs type:", type(envs))
    print("Single Action Space:", envs.single_action_space)
    print("Single Observation Space:", envs.single_observation_space)

    assert args.observation_mode in ['both', 'state', 'image']
    if args.observation_mode == "both":
        obs_shape = (envs.embedding_size,)
    elif args.observation_mode == "state":
        obs_shape = (envs.state_size,)
    elif args.observation_mode == "image":
        obs_shape = (envs.image_embedding_size,)

    agent = Agent(envs, obs_shape).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)

    # ALGO Logic: Storage setup
    # each obs is like {'image': {'rgb': (B,H,W,6), 'depth': (B,H,W,2)}, 'state': (B,D)}
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

     # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = process_obs_dict(next_obs, args.observation_mode)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    result = defaultdict(list) # yuan

    for iteration in range(1, args.num_iterations + 1):
        timeout_bonus = torch.zeros((args.num_steps, args.num_envs), device=device)
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_obs = process_obs_dict(next_obs, args.observation_mode)
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            for env_id, ended_ in enumerate(next_done):
                # we don't save the real next_obs if done, so we have to deal with it here
                if ended_:
                    terminal_obs = process_obs_dict(infos[env_id]["terminal_observation"], args.observation_mode)
                    with torch.no_grad():
                        terminal_value = agent.get_value(terminal_obs)
                    timeout_bonus[step, env_id] = args.gamma * terminal_value.item()
            result = collect_episode_info(infos, result) # yuan

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            rewards_ = rewards + timeout_bonus
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards_[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
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

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if len(result['return']) > 0:
            for k, v in result.items():
                writer.add_scalar(f"train/{k}", np.mean(v), global_step)
            result = defaultdict(list) # yuan
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

    envs.close()
    writer.close()
