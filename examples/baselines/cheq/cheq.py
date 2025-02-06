import os
import random
import time
from dataclasses import dataclass
from typing import Optional
<<<<<<< HEAD
import numpy as np

import gymnasium as gym
=======

import gymnasium as gym
import numpy as np
>>>>>>> 0ac0e35 (cheq implementation v1)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
import tqdm
import tyro

from torch.utils.tensorboard import SummaryWriter

from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils import gym_utils

<<<<<<< HEAD
=======

>>>>>>> 0ac0e35 (cheq implementation v1)
@dataclass
class CHEQArgs:
    # experiment
    exp_name: Optional[str] = None
    seed: int = 0
    torch_deterministic: bool = True
    track: bool = False
    wandb_project_name: str = "CHEQ"
    wandb_entity: Optional[str] = None

    # environment
    env_id: str = "PickCube-v1"
    num_envs: int = 16
    num_eval_envs: int = 4
    partial_reset: bool = False
    eval_partial_reset: bool = False
<<<<<<< HEAD
    control_mode: Optional[str] = "pd_joint_vel"
=======
    control_mode: Optional[str] = "pd_joint_delta_pos"
>>>>>>> 0ac0e35 (cheq implementation v1)
    reconfiguration_freq: Optional[int] = None
    eval_reconfiguration_freq: Optional[int] = 1
    obs_mode: str = "state"
    render_mode: str = "rgb_array"
    sim_backend: str = "gpu"

    # rollout
    total_timesteps: int = 1_000_000
    evaluate: bool = False
    eval_freq: int = 10_000
<<<<<<< HEAD
    num_steps: int = 400
    num_eval_steps: int = 400
    env_horizon: int = 400
=======
    num_eval_steps: int = 50
>>>>>>> 0ac0e35 (cheq implementation v1)
    capture_video: bool = False
    save_trajectory: bool = False
    save_model: bool = True
    checkpoint: Optional[str] = None

    # replay buffer
    buffer_size: int = 1_000_000
    buffer_device: str = "cuda"

    # training hyperparams
<<<<<<< HEAD
    gamma: float = 0.8
=======
    gamma: float = 0.99
>>>>>>> 0ac0e35 (cheq implementation v1)
    tau: float = 0.005
    policy_lr: float = 3e-4
    q_lr: float = 1e-3
    alpha: float = 0.2
    autotune: bool = True
    target_network_frequency: int = 1
    policy_frequency: int = 2
    batch_size: int = 256
    learning_starts: int = 10_000
    learning_starts_actor: Optional[int] = None
    start_random: bool = True
    grad_steps_per_iteration: int = 1
    log_freq: int = 1000

    # ensemble & bernoulli
    ensemble_size: int = 5
    bernoulli_mask_coeff: float = 1.0
    random_target: bool = True

    # prior
<<<<<<< HEAD
    prior_ckpt: Optional[str] = None  # path to .pt containing 'actor' dict (BC)
=======
    prior_ckpt: Optional[str] = None  # path to .pt containing 'actor' dict
>>>>>>> 0ac0e35 (cheq implementation v1)

    # cheq mixing
    uhigh: float = 0.15
    ulow: float = 0.03
    lam_high: float = 1.0
    lam_low: float = 0.2

<<<<<<< HEAD
LOG_STD_MAX = 2
LOG_STD_MIN = -5

class ActorWithLambda(nn.Module):
    """
    RL policy: receives (obs + 1D lambda) -> outputs action distribution.
=======

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class ActorWithLambda(nn.Module):
    """
    RL Actor that expects [state_dim + 1] as input (the last dim is λ).
>>>>>>> 0ac0e35 (cheq implementation v1)
    """
    def __init__(self, obs_dim_plus_1: int, act_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim_plus_1, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, act_dim)
        self.fc_logstd = nn.Linear(256, act_dim)

        self.action_scale = nn.Parameter(torch.ones(act_dim), requires_grad=False)
        self.action_bias  = nn.Parameter(torch.zeros(act_dim), requires_grad=False)

    def configure_action_scale(self, action_space: gym.spaces.Box):
<<<<<<< HEAD
        high, low = action_space.high, action_space.low
        scale = torch.tensor((high - low)/2.0, dtype=torch.float32)
        bias  = torch.tensor((high + low)/2.0, dtype=torch.float32)
=======
        h, l = action_space.high, action_space.low
        scale = torch.tensor((h - l)/2.0, dtype=torch.float32)
        bias  = torch.tensor((h + l)/2.0, dtype=torch.float32)
>>>>>>> 0ac0e35 (cheq implementation v1)
        self.action_scale.copy_(scale)
        self.action_bias.copy_(bias)

    def forward(self, obs_plus_lambda):
        x = F.relu(self.fc1(obs_plus_lambda))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
<<<<<<< HEAD

        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN)*(log_std + 1)
=======
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5*(LOG_STD_MAX - LOG_STD_MIN)*(log_std + 1)
>>>>>>> 0ac0e35 (cheq implementation v1)
        return mean, log_std

    def get_action(self, obs_plus_lambda, deterministic=False):
        mean, log_std = self(obs_plus_lambda)
        if deterministic:
            act = torch.tanh(mean)*self.action_scale + self.action_bias
            return act, None, act

        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()
        y = torch.tanh(z)
        act = y*self.action_scale + self.action_bias
<<<<<<< HEAD

        log_prob = dist.log_prob(z)
        log_prob -= torch.log(self.action_scale*(1 - y**2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        eval_act = torch.tanh(mean)*self.action_scale + self.action_bias
        return act, log_prob, eval_act

class SoftQNetwork(nn.Module):
=======
        log_prob = dist.log_prob(z)
        log_prob -= torch.log(self.action_scale*(1 - y**2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        eval_act = torch.tanh(mean)*self.action_scale + self.action_bias
        return act, log_prob, eval_act


class SoftQNetwork(nn.Module):
    """One Q-network expecting [obs_dim+1, action_dim]."""
>>>>>>> 0ac0e35 (cheq implementation v1)
    def __init__(self, obs_dim_plus_1: int, act_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim_plus_1 + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, obs_plus_lambda, act):
        x = torch.cat([obs_plus_lambda, act], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

<<<<<<< HEAD
class ActorPriorBC(nn.Module):
    """
    BC policy that expects only the plain obs (no lambda dimension).
    """
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
        )
=======

class ActorPriorNoLambda(nn.Module):
    """
    Prior policy (used a BC policy in this case): trained only on original 'state_dim' (no lambda).
    """
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, act_dim)
        self.fc_logstd = nn.Linear(256, act_dim)

>>>>>>> 0ac0e35 (cheq implementation v1)
        self.action_scale = nn.Parameter(torch.ones(act_dim), requires_grad=False)
        self.action_bias  = nn.Parameter(torch.zeros(act_dim), requires_grad=False)

    def configure_action_scale(self, action_space: gym.spaces.Box):
<<<<<<< HEAD
        high, low = action_space.high, action_space.low
        scale = torch.tensor((high - low)/2.0, dtype=torch.float32)
        bias  = torch.tensor((high + low)/2.0, dtype=torch.float32)
        self.action_scale.copy_(scale)
        self.action_bias.copy_(bias)

    def forward(self, obs):
        return self.net(obs)

    def get_action(self, obs, deterministic=True):
        # We'll do a simple tanh-squash for BC
        x = self.forward(obs)
        act = torch.tanh(x)*self.action_scale + self.action_bias
        return act, None, act

=======
        h, l = action_space.high, action_space.low
        scale = torch.tensor((h - l)/2.0, dtype=torch.float32)
        bias  = torch.tensor((h + l)/2.0, dtype=torch.float32)
        self.action_scale.copy_(scale)
        self.action_bias.copy_(bias)

    def forward(self, plain_obs):
        x = F.relu(self.fc1(plain_obs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5*(LOG_STD_MAX - LOG_STD_MIN)*(log_std + 1)
        return mean, log_std

    def get_action(self, plain_obs, deterministic=False):
        mean, log_std = self(plain_obs)
        if deterministic:
            act = torch.tanh(mean)*self.action_scale + self.action_bias
            return act, None, act
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()
        y = torch.tanh(z)
        act = y*self.action_scale + self.action_bias
        log_prob = dist.log_prob(z)
        log_prob -= torch.log(self.action_scale*(1 - y**2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        eval_act = torch.tanh(mean)*self.action_scale + self.action_bias
        return act, log_prob, eval_act


# ---------------------------------------------------------------------
# 3) REPLAY (Bernoulli)
# ---------------------------------------------------------------------
>>>>>>> 0ac0e35 (cheq implementation v1)
class ReplayBufferSample:
    def __init__(self, obs, next_obs, actions, rewards, dones, masks):
        self.obs = obs
        self.next_obs = next_obs
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.masks = masks

<<<<<<< HEAD
class BernoulliMaskReplayBuffer:
=======

class BernoulliMaskReplayBuffer:
    """Stores transitions with a random 0/1 mask for each ensemble Q."""
>>>>>>> 0ac0e35 (cheq implementation v1)
    def __init__(self, buffer_size, obs_dim_plus_1, act_dim, ensemble_size, p_masking, device):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False

        self.obs = torch.zeros((buffer_size, obs_dim_plus_1), device=device)
        self.next_obs = torch.zeros((buffer_size, obs_dim_plus_1), device=device)
        self.actions = torch.zeros((buffer_size, act_dim), device=device)
        self.rewards = torch.zeros((buffer_size, 1), device=device)
        self.dones   = torch.zeros((buffer_size, 1), device=device)
<<<<<<< HEAD

        # Mask for each ensemble head
        self.masks = torch.zeros((buffer_size, ensemble_size), device=device)
=======
        self.masks   = torch.zeros((buffer_size, ensemble_size), device=device)
>>>>>>> 0ac0e35 (cheq implementation v1)

        self.ensemble_size = ensemble_size
        self.p_masking = p_masking
        self.device = device

    def add(self, obs_aug, next_obs_aug, a_rl, rew, done):
        idx = self.pos
        self.obs[idx]      = obs_aug
        self.next_obs[idx] = next_obs_aug
        self.actions[idx]  = a_rl
        self.rewards[idx]  = rew
        self.dones[idx]    = done

        mask_i = torch.bernoulli(self.p_masking*torch.ones(self.ensemble_size, device=self.device))
        self.masks[idx] = mask_i

        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):
        high = self.buffer_size if self.full else self.pos
        idxs = torch.randint(0, high, (batch_size,), device=self.device)

        return ReplayBufferSample(
            obs=self.obs[idxs],
            next_obs=self.next_obs[idxs],
            actions=self.actions[idxs],
            rewards=self.rewards[idxs],
            dones=self.dones[idxs],
            masks=self.masks[idxs]
        )

<<<<<<< HEAD
def inject_weight_into_state(obs: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """
    lam is shape [N, 1], obs is shape [N, obs_dim].
    We cat them -> shape [N, obs_dim+1].
    """
    return torch.cat([obs, lam], dim=1)

def remove_lambda_dimension(obs_aug: torch.Tensor) -> torch.Tensor:
    return obs_aug[..., :-1]

def get_lambda(u: torch.Tensor, args: CHEQArgs):
    """
    Map uncertainty u to a mixture weight lambda in [lam_low, lam_high].
    u: shape [N]. Return shape [N,1].
    """
    lam_vals = torch.empty_like(u)
    lower_mask = (u <= args.ulow)
    upper_mask = (u >= args.uhigh)
    mid_mask   = ~(lower_mask | upper_mask)

    lam_vals[lower_mask] = args.lam_low
    lam_vals[upper_mask] = args.lam_high

    mid_u = u[mid_mask]
    frac = (mid_u - args.ulow)/(args.uhigh - args.ulow)  # in [0,1]
    lam_vals[mid_mask] = args.lam_low + frac*(args.lam_high - args.lam_low)

    return lam_vals.unsqueeze(1)  # shape [N,1]

def train_cheq(args: CHEQArgs):
    # Seeding
=======
# ---------------------------------------------------------------------
# 4) UTILS
# ---------------------------------------------------------------------
def inject_weight_into_state(s: torch.Tensor, lam: float) -> torch.Tensor:
    """
    s: [N, obs_dim], lam -> produce shape [N, obs_dim+1].
    """
    N = s.shape[0]
    lam_col = torch.full((N, 1), lam, dtype=s.dtype, device=s.device)
    return torch.cat([s, lam_col], dim=1)

def remove_lambda_dimension(obs_aug: torch.Tensor) -> torch.Tensor:
    """If obs_aug is [N, obs_dim+1], strip off the last dim => [N, obs_dim]."""
    return obs_aug[..., :-1]

# ---------------------------------------------------------------------
# 5) MAIN TRAINING
# ---------------------------------------------------------------------
def train_cheq(args: CHEQArgs):
    # A) Seeding & device
>>>>>>> 0ac0e35 (cheq implementation v1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True
<<<<<<< HEAD

=======
>>>>>>> 0ac0e35 (cheq implementation v1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = args.exp_name or "cheq_experiment"
    run_name = f"{args.env_id}__{run_name}__{args.seed}__{int(time.time())}"

<<<<<<< HEAD
    # Create vectorized train / eval env
=======
    # B) Create env
>>>>>>> 0ac0e35 (cheq implementation v1)
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        render_mode=args.render_mode,
        sim_backend=args.sim_backend,
        control_mode=args.control_mode,
        reconfiguration_freq=args.reconfiguration_freq
    )
<<<<<<< HEAD

    train_env = gym.make(
        args.env_id,
        robot_uids="xarm6_pandagripper",
        num_envs=args.num_envs if not args.evaluate else 1,
        max_episode_steps=args.env_horizon,
=======
    train_env = gym.make(
        args.env_id,
        num_envs=args.num_envs if not args.evaluate else 1,
>>>>>>> 0ac0e35 (cheq implementation v1)
        **env_kwargs
    )
    eval_env = gym.make(
        args.env_id,
<<<<<<< HEAD
        robot_uids="xarm6_pandagripper",
        num_envs=args.num_eval_envs,
        max_episode_steps=args.env_horizon,
        **env_kwargs
    )

    # Flatten
=======
        num_envs=args.num_eval_envs,
        reconfiguration_freq=args.eval_reconfiguration_freq,
        **env_kwargs
    )

>>>>>>> 0ac0e35 (cheq implementation v1)
    if isinstance(train_env.action_space, gym.spaces.Dict):
        train_env = FlattenActionSpaceWrapper(train_env)
        eval_env  = FlattenActionSpaceWrapper(eval_env)

<<<<<<< HEAD
    # Recording
=======
>>>>>>> 0ac0e35 (cheq implementation v1)
    if args.capture_video or args.save_trajectory:
        video_dir = f"runs/{run_name}/videos"
        if args.evaluate and args.checkpoint is not None:
            video_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        eval_env = RecordEpisode(
            eval_env,
            output_dir=video_dir,
            save_trajectory=args.save_trajectory,
            save_video=args.capture_video,
            trajectory_name="trajectory",
            max_steps_per_video=args.num_eval_steps,
            video_fps=30
        )

    train_env = ManiSkillVectorEnv(train_env, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
<<<<<<< HEAD
    eval_env  = ManiSkillVectorEnv(eval_env, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
=======
    eval_env  = ManiSkillVectorEnv(eval_env,  args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
>>>>>>> 0ac0e35 (cheq implementation v1)

    original_obs_dim = np.prod(train_env.single_observation_space.shape)
    obs_dim_plus_1   = original_obs_dim + 1
    act_dim          = np.prod(train_env.single_action_space.shape)

<<<<<<< HEAD
    # RL actor & ensemble Q
=======
    # C) RL Actor, Q ensemble
>>>>>>> 0ac0e35 (cheq implementation v1)
    actor = ActorWithLambda(obs_dim_plus_1, act_dim).to(device)
    actor.configure_action_scale(train_env.single_action_space)

    qfs = nn.ModuleList([SoftQNetwork(obs_dim_plus_1, act_dim).to(device) for _ in range(args.ensemble_size)])
    qfs_target = nn.ModuleList([SoftQNetwork(obs_dim_plus_1, act_dim).to(device) for _ in range(args.ensemble_size)])
    for i in range(args.ensemble_size):
        qfs_target[i].load_state_dict(qfs[i].state_dict())

    q_optimizer = optim.Adam([p for qf in qfs for p in qf.parameters()], lr=args.q_lr)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)

<<<<<<< HEAD
    if args.autotune:
        target_entropy = -float(act_dim)
=======
    # alpha
    if args.autotune:
        target_entropy = -float(act_dim)  # typical
>>>>>>> 0ac0e35 (cheq implementation v1)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha
        log_alpha = None

<<<<<<< HEAD
    # Load BC prior if provided
    prior_actor = None
    if args.prior_ckpt:
        prior_actor = ActorPriorBC(original_obs_dim, act_dim).to(device)
        prior_actor.configure_action_scale(train_env.single_action_space)
        ckpt = torch.load(args.prior_ckpt, map_location=device)
        if "actor" in ckpt:
            try:
                prior_actor.load_state_dict(ckpt["actor"], strict=True)
            except RuntimeError:
                prior_actor.load_state_dict(ckpt["actor"], strict=False)
            print(f"[CHEQ] Loaded BC prior from {args.prior_ckpt}")
        else:
            print(f"[CHEQ] No 'actor' key found in {args.prior_ckpt}")

    # Replay buffer
=======
    # D) Prior actor if provided
    prior_actor = None
    if args.prior_ckpt:
        prior_actor = ActorPriorNoLambda(original_obs_dim, act_dim).to(device)
        prior_actor.configure_action_scale(train_env.single_action_space)
        prior_ckpt = torch.load(args.prior_ckpt, map_location=device)
        prior_actor.load_state_dict(prior_ckpt['actor'])
        print(f"[CHEQ] Loaded prior from {args.prior_ckpt}")

    # E) Replay buffer
>>>>>>> 0ac0e35 (cheq implementation v1)
    rb = BernoulliMaskReplayBuffer(
        buffer_size=args.buffer_size,
        obs_dim_plus_1=obs_dim_plus_1,
        act_dim=act_dim,
        ensemble_size=args.ensemble_size,
        p_masking=args.bernoulli_mask_coeff,
        device=device
    )

<<<<<<< HEAD
    @torch.no_grad()
    def compute_uncertainty(obs_plus_lambda, action_rl):
        qs = [qf(obs_plus_lambda, action_rl) for qf in qfs]
        qs_cat = torch.cat(qs, dim=1)
        return qs_cat.std(dim=1)  # shape [N]

    # Evaluation
    def do_evaluation(step_num: int):
        actor.eval()
        if prior_actor is not None:
            prior_actor.eval()

        eval_obs, _ = eval_env.reset()
        ret = torch.zeros(args.num_eval_envs, dtype=torch.float32, device=device)

        for _ in range(args.num_eval_steps):
            with torch.no_grad():
                lam_ones = torch.ones((args.num_eval_envs, 1), device=device)
                obs_aug_eval = inject_weight_into_state(eval_obs, lam_ones)
                pi_eval_act, _, _ = actor.get_action(obs_aug_eval, deterministic=True)

                stdvals = compute_uncertainty(obs_aug_eval, pi_eval_act)  # shape [N]
                lam_eval = get_lambda(stdvals, args)                      # shape [N,1]

                if prior_actor is not None:
                    plain_obs_eval = remove_lambda_dimension(obs_aug_eval)
                    bc_act, _, _ = prior_actor.get_action(plain_obs_eval, deterministic=True)
                else:
                    bc_act = torch.zeros_like(pi_eval_act)

                final_eval_action = lam_eval * pi_eval_act + (1 - lam_eval) * bc_act

            next_obs, rew, done, trunc, infos = eval_env.step(final_eval_action)
            ret += rew
            eval_obs = next_obs

        mean_ret = ret.mean().item()
        # Log to TB & wandb
        if writer:
            writer.add_scalar("eval/return", mean_ret, step_num)
        if args.track:
            wandb.log({"eval/return": mean_ret}, step=step_num)

        actor.train()
        if prior_actor is not None:
            prior_actor.train()
        return mean_ret

    # Tracking
=======
    # F) Helpers: uncertainty & lambda
    def compute_uncertainty(obs_aug, pi_act):
        with torch.no_grad():
            qs = [qf(obs_aug, pi_act) for qf in qfs]
            qs_cat = torch.cat(qs, dim=1)  # [N, ensemble_size]
            return qs_cat.std(dim=1)

    def get_lambda(u):
        if u <= args.ulow:
            return args.lam_low
        elif u >= args.uhigh:
            return args.lam_high
        else:
            frac = (u - args.ulow)/(args.uhigh - args.ulow)
            return args.lam_low + frac*(args.lam_high - args.lam_low)

    # G) Logging
>>>>>>> 0ac0e35 (cheq implementation v1)
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name
        )
    writer = SummaryWriter(f"runs/{run_name}") if not args.evaluate else None

<<<<<<< HEAD
    obs, _ = train_env.reset(seed=args.seed)
=======
    # H) Main training loop
    obs_np, _ = train_env.reset(seed=args.seed)
    obs_t = torch.from_numpy(obs_np).float().to(device)

>>>>>>> 0ac0e35 (cheq implementation v1)
    global_step = 0
    start_time = time.time()
    learning_starts_actor = args.learning_starts_actor or args.learning_starts

<<<<<<< HEAD
    # Main Loop
    for step in tqdm.trange(args.total_timesteps // args.num_envs):
        global_step += args.num_envs

        # 1) RL actor or random
        if global_step >= args.learning_starts or not args.start_random:
            lam_ones = torch.ones((args.num_envs, 1), device=device)
            obs_aug_input = torch.cat([obs, lam_ones], dim=1)
            with torch.no_grad():
                pi_action, _, _ = actor.get_action(obs_aug_input, deterministic=False)
        else:
            rand_action_np = train_env.action_space.sample()  # shape (num_envs, act_dim)
            pi_action = torch.tensor(rand_action_np, dtype=torch.float32, device=device)

        # 2) BC prior
        if prior_actor is not None:
            with torch.no_grad():
                bc_action, _, _ = prior_actor.get_action(obs, deterministic=True)
        else:
            bc_action = torch.zeros_like(pi_action, device=device)

        # 3) Uncertainty => lam
        if global_step < args.learning_starts:
            lam_vec = torch.zeros((args.num_envs, 1), device=device)  # pure BC at start
        else:
            with torch.no_grad():
                stdvals = compute_uncertainty(obs_aug_input, pi_action)
                lam_vec = get_lambda(stdvals, args)

        # 4) Final action mix
        final_action = lam_vec * pi_action + (1 - lam_vec) * bc_action

        # 5) Step environment
        next_obs, reward, done, trunc, infos = train_env.step(final_action)

        # 6) Store transitions
        obs_aug_store     = torch.cat([obs, lam_vec], dim=1)
        next_obs_aug_store= torch.cat([next_obs, lam_vec], dim=1)

        for i in range(args.num_envs):
            rb.add(
                obs_aug_store[i],
                next_obs_aug_store[i],
                pi_action[i],
                reward[i].unsqueeze(0),
                (done[i] | trunc[i]).float().unsqueeze(0)
            )

        obs = next_obs

        # -- Logging immediate rollout info --
        mean_rew = reward.mean().item()
        lam_mean = lam_vec.mean().item()
        if args.track:
            wandb.log({"rollout/mean_reward": mean_rew,
                       "rollout/lambda_mean":   lam_mean},
                      step=global_step)
        # We'll only do TB logging occasionally:
        if writer and (global_step % args.log_freq == 0):
            writer.add_scalar("rollout/mean_reward", mean_rew, global_step)
            writer.add_scalar("rollout/lambda_mean", lam_mean, global_step)
            writer.add_scalar("time/elapsed", time.time() - start_time, global_step)

        # 7) Training
        if global_step >= args.learning_starts:
            for _ in range(args.grad_steps_per_iteration):
                batch = rb.sample(args.batch_size)

                # Q update
                with torch.no_grad():
                    next_pi, next_logp, _ = actor.get_action(batch.next_obs)
                    q_tgts = [qf_t(batch.next_obs, next_pi) for qf_t in qfs_target]
                    q_cat = torch.cat(q_tgts, dim=1)
                    min_qt = q_cat.min(dim=1, keepdim=True).values

                    y = batch.rewards.flatten() \
                      + (1 - batch.dones.flatten()) * args.gamma \
                        * (min_qt.view(-1) - alpha*next_logp.view(-1))
=======
    def do_evaluation(step_num: int):
        """Simple evaluation loop with the same CHEQ logic (uncertainty-based λ).
           We run parallel for `args.num_eval_steps` steps, log average return."""
        actor.eval()
        if prior_actor:
            prior_actor.eval()

        eval_obs_np, _ = eval_env.reset()
        eval_obs_t = torch.from_numpy(eval_obs_np).float().to(device)
        # We track returns for each parallel env
        ret = np.zeros(args.num_eval_envs, dtype=np.float32)
        # For simplicity, we do a fixed number of steps
        for _eval_step in range(args.num_eval_steps):
            # measure RL action
            obs_aug_eval = inject_weight_into_state(eval_obs_t, lam=1.0)  # temporarily 1.0 for unc. measure
            with torch.no_grad():
                pi_eval_act, _, _ = actor.get_action(obs_aug_eval, deterministic=True)
            # measure uncertainty => real lam
            stdvals = compute_uncertainty(obs_aug_eval, pi_eval_act)
            lam_eval = get_lambda(stdvals.mean().item())  # or do per-env
            # final action
            prior_act = None
            if prior_actor is not None:
                pure_obs_eval = remove_lambda_dimension(obs_aug_eval)
                prior_act, _, _ = prior_actor.get_action(pure_obs_eval, deterministic=True)
            else:
                prior_act = torch.zeros_like(pi_eval_act)
            final_eval_action = lam_eval*pi_eval_act + (1 - lam_eval)*prior_act
            # step
            next_eval_obs_np, rew_np, done_np, trunc_np, infos = eval_env.step(final_eval_action)
            # accumulate return
            ret += rew_np
            eval_obs_t = torch.from_numpy(next_eval_obs_np).float().to(device)

        mean_ret = ret.mean()
        if writer:
            writer.add_scalar("eval/return", mean_ret, step_num)
        if args.track:
            wandb.log({"eval/return": mean_ret}, step=step_num)

        actor.train()
        if prior_actor:
            prior_actor.train()
        return mean_ret

    for step in tqdm.trange(args.total_timesteps // args.num_envs):
        global_step += args.num_envs

        # 1) compute RL action
        lam_temp = 1.0 if (global_step >= args.learning_starts) else 0.0
        obs_aug = inject_weight_into_state(obs_t, lam_temp)
        if global_step >= args.learning_starts or not args.start_random:
            with torch.no_grad():
                pi_action, _, _ = actor.get_action(obs_aug, deterministic=False)
        else:
            pi_action = torch.tensor(train_env.action_space.sample(), dtype=torch.float32, device=device)

        # 2) prior action
        if prior_actor is not None:
            plain_s = remove_lambda_dimension(obs_aug)
            with torch.no_grad():
                prior_act, _, _ = prior_actor.get_action(plain_s, deterministic=True)
        else:
            prior_act = torch.zeros_like(pi_action, device=device)

        # 3) measure uncertainty => real lam
        if global_step < args.learning_starts:
            lam_actual = 0.0
        else:
            stdvals = compute_uncertainty(obs_aug, pi_action)
            lam_actual = get_lambda(stdvals.mean().item())

        # 4) final action
        final_action = lam_actual*pi_action + (1 - lam_actual)*prior_act

        # step environment
        next_obs_np, rew_np, done_np, trunc_np, infos = train_env.step(final_action)
        next_obs_t = torch.from_numpy(next_obs_np).float().to(device)

        # store in replay
        obs_aug_true = inject_weight_into_state(obs_t, lam_actual)
        next_obs_aug_true = inject_weight_into_state(next_obs_t, lam_actual)
        rb.add(
            obs_aug_true[0],
            next_obs_aug_true[0],
            pi_action[0],
            torch.tensor([rew_np[0]], dtype=torch.float32, device=device),
            torch.tensor([done_np[0] or trunc_np[0]], dtype=torch.float32, device=device)
        )
        obs_t = next_obs_t

        # 5) train updates
        if global_step >= args.learning_starts:
            for _ in range(args.grad_steps_per_iteration):
                batch = rb.sample(args.batch_size)
                # Critic update
                with torch.no_grad():
                    next_pi, next_logp, _ = actor.get_action(batch.next_obs)
                    q_targets = []
                    for qtarg in qfs_target:
                        q_targets.append(qtarg(batch.next_obs, next_pi))
                    q_cat = torch.cat(q_targets, dim=1)
                    min_qt = q_cat.min(dim=1, keepdim=True).values
                    y = batch.rewards.flatten() + (1 - batch.dones.flatten())*args.gamma*(min_qt.view(-1) - alpha*next_logp.view(-1))
>>>>>>> 0ac0e35 (cheq implementation v1)

                q_loss_total = 0.0
                for j, qf_main in enumerate(qfs):
                    q_est = qf_main(batch.obs, batch.actions).view(-1)
                    mask_j = batch.masks[:, j]
                    mse_j = (q_est - y)**2 * mask_j
                    q_loss_j = mse_j.sum()/(mask_j.sum() + 1e-6)
                    q_loss_total += q_loss_j
<<<<<<< HEAD

=======
>>>>>>> 0ac0e35 (cheq implementation v1)
                q_optimizer.zero_grad()
                q_loss_total.backward()
                q_optimizer.step()

<<<<<<< HEAD
                # Optionally log Q-loss each step
                q_loss_val = q_loss_total.item()
                if args.track:
                    wandb.log({"train/q_loss": q_loss_val}, step=global_step)
                if writer and (global_step % args.log_freq == 0):
                    writer.add_scalar("train/q_loss", q_loss_val, global_step)

                # Polyak update
                if global_step % args.target_network_frequency == 0:
                    for q_main, q_targ in zip(qfs, qfs_target):
                        for p_m, p_t in zip(q_main.parameters(), q_targ.parameters()):
                            p_t.data.copy_(args.tau*p_m.data + (1 - args.tau)*p_t.data)

            # Actor update
            if global_step >= learning_starts_actor and (global_step % args.policy_frequency == 0):
                data2 = rb.sample(args.batch_size)
                pi2, logp2, _ = actor.get_action(data2.obs)

                all_q_vals = [qf_m(data2.obs, pi2) for qf_m in qfs]
                cat_q_vals = torch.cat(all_q_vals, dim=1)
                min_q_pi = cat_q_vals.min(dim=1, keepdim=True).values
                policy_loss = (alpha * logp2 - min_q_pi).mean()
=======
                if global_step % args.target_network_frequency == 0:
                    for q_main, q_targ in zip(qfs, qfs_target):
                        for p_m, p_t in zip(q_main.parameters(), q_targ.parameters()):
                            p_t.data.copy_(args.tau*p_m.data + (1-args.tau)*p_t.data)

            # actor update
            if global_step >= (learning_starts_actor) and (global_step % args.policy_frequency == 0):
                data2 = rb.sample(args.batch_size)
                pi2, logp2, _ = actor.get_action(data2.obs)
                all_q_vals = []
                for qf_m in qfs:
                    all_q_vals.append(qf_m(data2.obs, pi2))
                cat_q_vals = torch.cat(all_q_vals, dim=1)
                min_q_pi = cat_q_vals.min(dim=1, keepdim=True).values
                policy_loss = (alpha*logp2 - min_q_pi).mean()
>>>>>>> 0ac0e35 (cheq implementation v1)

                actor_optimizer.zero_grad()
                policy_loss.backward()
                actor_optimizer.step()

<<<<<<< HEAD
                # Log policy loss
                policy_loss_val = policy_loss.item()
                if args.track:
                    wandb.log({"train/policy_loss": policy_loss_val}, step=global_step)
                if writer and (global_step % args.log_freq == 0):
                    writer.add_scalar("train/policy_loss", policy_loss_val, global_step)

                # alpha autotune
=======
>>>>>>> 0ac0e35 (cheq implementation v1)
                if args.autotune:
                    with torch.no_grad():
                        _, logp3, _ = actor.get_action(data2.obs)
                    alpha_loss = -(log_alpha.exp()*(logp3 + target_entropy)).mean()
                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

<<<<<<< HEAD
                    # Log alpha
                    if args.track:
                        wandb.log({"train/alpha": alpha}, step=global_step)
                    if writer and (global_step % args.log_freq == 0):
                        writer.add_scalar("train/alpha", alpha, global_step)

        # Evaluate
        if (not args.evaluate) and (global_step % args.eval_freq == 0):
            eval_ret = do_evaluation(global_step)
            print(f"[Eval @ Step {global_step}] Return: {eval_ret:.3f}")

        if global_step >= args.total_timesteps:
            break

    # Save final
    if not args.evaluate and args.save_model:
        os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
        ckpt_path = f"runs/{run_name}/checkpoints/cheq_final.pt"
        torch.save({"actor": actor.state_dict()}, ckpt_path)
=======
        # 6) logging
        if writer is not None and (global_step % args.log_freq == 0):
            writer.add_scalar("rollout/lambda", lam_actual, global_step)
            writer.add_scalar("time/elapsed", time.time() - start_time, global_step)
            if global_step >= args.learning_starts:
                writer.add_scalar("losses/q_loss_total", q_loss_total.item(), global_step)
                if args.autotune:
                    writer.add_scalar("alpha/value", alpha, global_step)

        # 7) Evaluate if needed
        if (not args.evaluate) and (global_step % args.eval_freq == 0):
            eval_return = do_evaluation(global_step)
            print(f"[Eval @ Step {global_step}] Return: {eval_return:.3f}")

        # end condition
        if global_step >= args.total_timesteps:
            break

    # 8) final save
    if not args.evaluate and args.save_model:
        os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
        ckpt_path = f"runs/{run_name}/checkpoints/cheq_final.pt"
        torch.save({
            'actor': actor.state_dict(),
            # optionally qfs, alpha, etc.
        }, ckpt_path)
>>>>>>> 0ac0e35 (cheq implementation v1)
        print(f"[CHEQ] Saved final model to {ckpt_path}")

    train_env.close()
    eval_env.close()
    if writer:
        writer.close()
    if args.track:
        wandb.finish()

<<<<<<< HEAD
=======
# 6) Entry point
>>>>>>> 0ac0e35 (cheq implementation v1)
if __name__ == "__main__":
    args = tyro.cli(CHEQArgs)
    train_cheq(args)
