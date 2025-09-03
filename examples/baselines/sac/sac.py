import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import tyro
from torch.utils.tensorboard import SummaryWriter

import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: str = "SAC"
    """the group of the run for wandb"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_trajectory: bool = False
    """whether to save trajectory data into the `videos` folder"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    log_freq: int = 50_000
    """logging frequency in terms of environment steps"""

    # Environment specific arguments
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    env_vectorization: str = "gpu"
    """the type of environment vectorization to use"""
    num_envs: int = 1024
    """the number of parallel environments"""
    num_eval_envs: int = 16
    """the number of parallel evaluation environments"""
    partial_reset: bool = False
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    eval_freq: int = 500_000
    """evaluation frequency in terms of global step"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""

    # Algorithm specific arguments
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""

    buffer_size: int = 1_000_000
    """the replay memory buffer size"""
    buffer_device: str = "cuda"
    """where the replay buffer is stored. Can be 'cpu' or 'cuda' for GPU"""
    batch_size: int = 4096
    """the batch size of sample from the replay memory"""
    learning_starts: int = 4_000
    """timestep to start learning"""

    obs_rms: bool = False
    """whether to use observation normalization"""

    steps_per_env_per_iteration: int = 1
    """the number of steps each parallel env takes per iteration"""
    grad_steps_per_iteration: int = 20
    """the number of gradient updates per iteration"""
    policy_frequency: int = 1
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 2  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""

    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    actor_state_projection_dim: Optional[int] = None
    """the state projection dimension of the actor network (if None, no projection is used)"""

    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    q_state_projection_dim: Optional[int] = None
    """the state projection dimension of the Q network (if None, no projection is used)"""
    q_layer_norm: bool = True
    """whether to use layer normalization in the Q network"""
    q_dropout: Optional[float] = None
    """the dropout rate of the Q network"""
    min_q: int = 2
    """number of Q target networks to get the min over"""
    num_q: int = 2
    """total number of Q networks"""

    tau: float = 0.005
    """target smoothing coefficient"""
    alpha: float = 1.0
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    alpha_lr: float = 3e-4
    """the learning rate of the entropy coefficient"""

    gamma: float = 0.8
    """the discount factor gamma"""
    bootstrap_at_done: str = "always"
    """the bootstrap method to use when a done signal is received. Can be 'always' or 'never'"""

    log_std_min: float = -5.0
    """the minimum log standard deviation"""
    log_std_max: float = 2.0
    """the maximum log standard deviation"""


@dataclass
class ReplayBufferSample:
    obs: torch.Tensor
    next_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(
        self,
        env,
        num_envs: int,
        buffer_size: int,
        storage_device: torch.device,
        sample_device: torch.device,
    ):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.num_envs = num_envs
        self.storage_device = storage_device
        self.sample_device = sample_device
        self.per_env_buffer_size = buffer_size // num_envs
        self.obs = torch.zeros(
            (self.per_env_buffer_size, self.num_envs)
            + env.single_observation_space.shape
        ).to(storage_device)
        self.next_obs = torch.zeros(
            (self.per_env_buffer_size, self.num_envs)
            + env.single_observation_space.shape
        ).to(storage_device)
        self.actions = torch.zeros(
            (self.per_env_buffer_size, self.num_envs) + env.single_action_space.shape
        ).to(storage_device)
        self.logprobs = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(
            storage_device
        )
        self.rewards = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(
            storage_device
        )
        self.dones = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(
            storage_device
        )
        self.values = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(
            storage_device
        )

    def add(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ):
        if self.storage_device == torch.device("cpu"):
            obs = obs.cpu()
            next_obs = next_obs.cpu()
            action = action.cpu()
            reward = reward.cpu()
            done = done.cpu()

        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs

        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos += 1
        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int):
        if self.full:
            batch_inds = torch.randint(0, self.per_env_buffer_size, size=(batch_size,))
        else:
            batch_inds = torch.randint(0, self.pos, size=(batch_size,))
        env_inds = torch.randint(0, self.num_envs, size=(batch_size,))
        return ReplayBufferSample(
            obs=self.obs[batch_inds, env_inds].to(self.sample_device),
            next_obs=self.next_obs[batch_inds, env_inds].to(self.sample_device),
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones=self.dones[batch_inds, env_inds].to(self.sample_device),
        )


# ALGO LOGIC: initialize agent here:
class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=(), device="cuda"):
        self.device = device
        self.mean = torch.zeros(shape, device=self.device)
        self.var = torch.ones(shape, device=self.device)
        self.epsilon = epsilon
        self.count = epsilon

    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def normalize(self, x):
        out = (x - self.mean) / torch.sqrt(self.var + self.epsilon)
        return out

    def unnormalize(self, x):
        out = x * torch.sqrt(self.var + self.epsilon) + self.mean
        return out

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def state_dict(self):
        return dict(mean=self.mean, var=self.var, epsilon=self.epsilon)

    def load_state_dict(self, info):
        self.mean = info["mean"]
        self.var = info["var"]
        self.count = info["count"]


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class RLProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.projection = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.Tanh()
        )
        self.out_dim = out_dim
        self.apply(weight_init)

    def forward(self, x):
        return self.projection(x)


class SoftQNetwork(nn.Module):
    def __init__(
        self,
        single_obs_space: gym.spaces.Box,
        single_act_space: gym.spaces.Box,
        hidden_dims: list[int] = [256, 256, 256],
        state_projection_dim: Optional[int] = None,
        layer_norm: bool = True,
        dropout: Optional[float] = None,
    ):
        super().__init__()

        if state_projection_dim is not None:
            self.state_projection = RLProjection(
                math.prod(single_obs_space.shape), state_projection_dim
            )
            layer_dims = (
                [state_projection_dim + math.prod(single_act_space.shape)]
                + hidden_dims
                + [1]
            )
        else:
            self.state_projection = None
            layer_dims = (
                [math.prod(single_obs_space.shape) + math.prod(single_act_space.shape)]
                + hidden_dims
                + [1]
            )

        layers = []
        for i, (in_d, out_d) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            layers.append(nn.Linear(in_d, out_d))
            if i != len(layer_dims[:-1]) - 1:
                if dropout is not None and dropout > 0:
                    layers.append(nn.Dropout(dropout))
                if layer_norm:
                    layers.append(nn.LayerNorm(out_d))
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

        self.apply(weight_init)

    def forward(self, x, a):
        if self.state_projection is not None:
            x = self.state_projection(x)
        x = torch.cat([x, a], 1)
        return self.net(x)


class Actor(nn.Module):
    def __init__(
        self,
        single_obs_space: gym.spaces.Box,
        single_act_space: gym.spaces.Box,
        hidden_dims: list[int] = [256, 256, 256],
        state_projection_dim: Optional[int] = None,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()

        if state_projection_dim is not None:
            self.state_projection = RLProjection(
                math.prod(single_obs_space.shape), state_projection_dim
            )
            layer_dims = [state_projection_dim] + hidden_dims
        else:
            self.state_projection = None
            layer_dims = [math.prod(single_obs_space.shape)] + hidden_dims

        layers = []
        for in_d, out_d in zip(layer_dims[:-1], layer_dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*layers)
        self.fc_mean = nn.Linear(layer_dims[-1], math.prod(single_act_space.shape))
        self.fc_logstd = nn.Linear(layer_dims[-1], math.prod(single_act_space.shape))
        # action rescaling
        h, l = single_act_space.high, single_act_space.low
        self.register_buffer(
            "action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32)
        )
        # will be saved in the state_dict

        # don't need to save these values
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max

        self.apply(weight_init)

    def forward(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self._log_std_min + 0.5 * (self._log_std_max - self._log_std_min) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_eval_action(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)


class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        self.writer.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    ####### Environment setup #######
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="gpu")
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    envs = gym.make(
        args.env_id,
        num_envs=args.num_envs if not args.evaluate else 1,
        reconfiguration_freq=args.reconfiguration_freq,
        **env_kwargs,
    )
    eval_envs = gym.make(
        args.env_id,
        num_envs=args.num_eval_envs,
        reconfiguration_freq=args.eval_reconfiguration_freq,
        human_render_camera_configs=dict(shader_pack="default"),
        **env_kwargs,
    )
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video or args.save_trajectory:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval trajectories/videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = (
                lambda x: (x // args.num_steps) % args.save_train_video_freq == 0
            )
            envs = RecordEpisode(
                envs,
                output_dir=f"runs/{run_name}/train_videos",
                save_trajectory=False,
                save_video_trigger=save_video_trigger,
                max_steps_per_video=args.num_steps,
                video_fps=30,
            )
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=eval_output_dir,
            save_trajectory=args.save_trajectory,
            save_video=args.capture_video,
            trajectory_name="trajectory",
            max_steps_per_video=args.num_eval_steps,
            video_fps=30,
        )
    envs = ManiSkillVectorEnv(
        envs,
        args.num_envs,
        ignore_terminations=not args.partial_reset,
        record_metrics=True,
    )
    eval_envs = ManiSkillVectorEnv(
        eval_envs,
        args.num_eval_envs,
        ignore_terminations=not args.eval_partial_reset,
        record_metrics=True,
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb

            config = vars(args)
            config["algo"] = "sac"
            config["env_cfg"] = dict(
                **env_kwargs,
                num_envs=args.num_envs,
                env_id=args.env_id,
                reward_mode="normalized_dense",
                env_horizon=max_episode_steps,
                partial_reset=args.partial_reset,
            )
            config["eval_env_cfg"] = dict(
                **env_kwargs,
                num_envs=args.num_eval_envs,
                env_id=args.env_id,
                reward_mode="normalized_dense",
                env_horizon=max_episode_steps,
                partial_reset=False,
            )
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group=args.wandb_group,
                tags=[
                    "sac",
                    "walltime_efficient",
                    f"GPU:{torch.cuda.get_device_name()}",
                ],
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    max_action = float(envs.single_action_space.high[0])

    actor_kwargs = dict(
        single_obs_space=envs.single_observation_space,
        single_act_space=envs.single_action_space,
        hidden_dims=[256, 256, 256],
        state_projection_dim=args.actor_state_projection_dim,
        log_std_min=args.log_std_min,
        log_std_max=args.log_std_max,
    )
    actor = Actor(**actor_kwargs).to(device)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    qf_kwargs = dict(
        single_obs_space=envs.single_observation_space,
        single_act_space=envs.single_action_space,
        hidden_dims=[256, 256, 256],
        state_projection_dim=args.q_state_projection_dim,
        layer_norm=args.q_layer_norm,
        dropout=args.q_dropout,
    )
    qfs = [SoftQNetwork(**qf_kwargs).to(device) for _ in range(args.num_q)]
    qf_targets = [SoftQNetwork(**qf_kwargs).to(device) for _ in range(args.num_q)]

    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint)
        actor.load_state_dict(ckpt["actor"])
        for qf, state_dict in zip(qfs, ckpt["qfs"]):
            qf.load_state_dict(state_dict)

    for qf, qf_target in zip(qfs, qf_targets):
        qf_target.load_state_dict(qf.state_dict())
    q_opt_params = []
    for qf in qfs:
        q_opt_params += list(qf.parameters())
    q_optimizer = optim.Adam(q_opt_params, lr=args.q_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(
            torch.Tensor(envs.single_action_space.shape).to(device)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        if args.checkpoint is not None:
            log_alpha[:] = ckpt["log_alpha"]
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)
    else:
        alpha = args.alpha

    if args.obs_rms:
        obs_rms = RunningMeanStd(
            shape=envs.single_observation_space.shape, device=device
        )
        if args.checkpoint is not None:
            obs_rms.load_state_dict(ckpt["obs_rms"])

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        env=envs,
        num_envs=args.num_envs,
        buffer_size=args.buffer_size,
        storage_device=torch.device(args.buffer_device),
        sample_device=device,
    )

    # TRY NOT TO MODIFY: start the game
    obs, info = envs.reset(
        seed=args.seed
    )  # in Gymnasium, seed is given to reset() instead of seed()
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    global_step = 0
    global_update = 0
    learning_has_started = False

    global_steps_per_iteration = args.num_envs * (args.steps_per_env_per_iteration)
    pbar = tqdm.tqdm(range(args.total_timesteps))
    cumulative_times = defaultdict(float)

    for iteration in range(
        math.ceil(args.total_timesteps / global_steps_per_iteration)
    ):
        with torch.inference_mode():
            if (
                args.eval_freq > 0
                and (global_step - global_steps_per_iteration) // args.eval_freq
                < global_step // args.eval_freq
            ):
                # evaluate
                actor.eval()
                stime = time.perf_counter()
                eval_obs, _ = eval_envs.reset()
                eval_metrics = defaultdict(list)
                num_episodes = 0
                for _ in range(args.num_eval_steps):
                    if args.obs_rms:
                        eval_action = actor.get_eval_action(obs_rms.normalize(eval_obs))
                    else:
                        eval_action = actor.get_eval_action(eval_obs)
                    (
                        eval_obs,
                        eval_rew,
                        eval_terminations,
                        eval_truncations,
                        eval_infos,
                    ) = eval_envs.step(eval_action)
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
                eval_metrics_mean = {}
                for k, v in eval_metrics.items():
                    mean = torch.stack(v).float().mean()
                    eval_metrics_mean[k] = mean
                    if logger is not None:
                        logger.add_scalar(f"eval/{k}", mean, global_step)
                pbar.set_description(
                    f"success_once: {eval_metrics_mean['success_once']:.2f}, "
                    f"return: {eval_metrics_mean['return']:.2f}",
                    refresh=False,
                )
                if logger is not None:
                    eval_time = time.perf_counter() - stime
                    cumulative_times["eval_time"] += eval_time
                    logger.add_scalar("time/eval_time", eval_time, global_step)
                if args.evaluate:
                    break
                actor.train()

                if args.save_model:
                    model_path = f"runs/{run_name}/ckpt_{global_step}.pt"
                    _save_dict = {
                        "actor": actor.state_dict(),
                        "qfs": [qf_target.state_dict() for qf_target in qf_targets],
                        "log_alpha": log_alpha,
                    }
                    if args.obs_rms:
                        _save_dict["obs_rms"] = obs_rms.state_dict()
                    torch.save(
                        _save_dict,
                        model_path,
                    )
                    print(f"model saved to {model_path}")

            # Collect samples from environemnts
            rollout_time = time.perf_counter()

            for local_step in range(args.steps_per_env_per_iteration):
                global_step += 1 * args.num_envs

                # ALGO LOGIC: put action logic here
                if args.obs_rms:
                    obs_rms.update(obs)
                if not learning_has_started:
                    actions = torch.tensor(
                        envs.action_space.sample(), dtype=torch.float32, device=device
                    )
                else:
                    if args.obs_rms:
                        actions, _, _ = actor.get_action(obs_rms.normalize(obs))
                    else:
                        actions, _, _ = actor.get_action(obs)
                    actions = actions.detach()

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, rewards, terminations, truncations, infos = envs.step(actions)
                real_next_obs = next_obs.clone()
                if args.bootstrap_at_done == "never":
                    need_final_obs = torch.ones_like(terminations, dtype=torch.bool)
                    stop_bootstrap = (
                        truncations | terminations
                    )  # always stop bootstrap when episode ends
                else:
                    if args.bootstrap_at_done == "always":
                        need_final_obs = (
                            truncations | terminations
                        )  # always need final obs when episode ends
                        stop_bootstrap = torch.zeros_like(
                            terminations, dtype=torch.bool
                        )  # never stop bootstrap
                    else:  # bootstrap at truncated
                        need_final_obs = truncations & (
                            ~terminations
                        )  # only need final obs when truncated and not terminated
                        stop_bootstrap = terminations  # only stop bootstrap when terminated, don't stop when truncated
                if "final_info" in infos:
                    final_info = infos["final_info"]
                    done_mask = infos["_final_info"]
                    real_next_obs[need_final_obs] = infos["final_observation"][
                        need_final_obs
                    ]
                    for k, v in final_info["episode"].items():
                        logger.add_scalar(
                            f"train/{k}", v[done_mask].float().mean(), global_step
                        )

                rb.add(obs, real_next_obs, actions, rewards, stop_bootstrap)

                # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
                obs = next_obs
            rollout_time = time.perf_counter() - rollout_time
            cumulative_times["rollout_time"] += rollout_time
            pbar.update(global_steps_per_iteration)

        # ALGO LOGIC: training.
        if global_step < args.learning_starts:
            continue

        update_time = time.perf_counter()
        num_grad_steps = (
            (args.grad_steps_per_iteration * iteration)
            if not learning_has_started
            else args.grad_steps_per_iteration
        )
        learning_has_started = True
        for local_update in range(args.grad_steps_per_iteration):
            global_update += 1
            data = rb.sample(args.batch_size)
            if args.obs_rms:
                data.obs = obs_rms.normalize(data.obs)
                data.next_obs = obs_rms.normalize(data.next_obs)

            # update the value networks
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(
                    data.next_obs
                )
                min_qf_next_target, _ = torch.min(
                    torch.cat(
                        [
                            qf_target(data.next_obs, next_state_actions)
                            for qf_target in random.sample(qf_targets, k=args.min_q)
                        ],
                        dim=1,
                    ),
                    dim=1,
                    keepdim=True,
                )
                min_qf_next_target -= alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * args.gamma * (min_qf_next_target).view(-1)
                # data.dones is "stop_bootstrap", which is computed earlier according to args.bootstrap_at_done

            qf_a_values = torch.cat([qf(data.obs, data.actions) for qf in qfs], dim=1)
            qf_loss = F.mse_loss(
                qf_a_values,
                next_q_value.unsqueeze(1).repeat(1, args.num_q),
                reduction="sum",
            )

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # update the policy network
            if (
                global_update % args.policy_frequency == 0
            ):  # TD 3 Delayed update support
                pi, log_pi, _ = actor.get_action(data.obs)
                qfs_pi = torch.cat([qf(data.obs, pi) for qf in qfs], dim=1)
                mean_qf_pi = torch.mean(qfs_pi, dim=1, keepdim=True)
                actor_loss = ((alpha * log_pi) - mean_qf_pi).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(data.obs)
                    # if args.correct_alpha:
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                    # else:
                    #     alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()
                    # log_alpha has a legacy reason: https://github.com/rail-berkeley/softlearning/issues/136#issuecomment-619535356

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # update the target networks
            if global_update % args.target_network_frequency == 0:
                for qf, qf_target in zip(qfs, qf_targets):
                    for param, target_param in zip(
                        qf.parameters(), qf_target.parameters()
                    ):
                        target_param.data.copy_(
                            args.tau * param.data + (1 - args.tau) * target_param.data
                        )
        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        # Log training-related data
        if (
            global_step - global_steps_per_iteration
        ) // args.log_freq < global_step // args.log_freq:
            logger.add_scalar(
                "losses/qf_mean_values",
                qf_a_values.mean().item(),
                global_step,
            )
            logger.add_scalar(
                "losses/qf_mean_loss", qf_loss.item() / args.num_q, global_step
            )
            logger.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            logger.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            logger.add_scalar("losses/alpha", alpha, global_step)
            logger.add_scalar("time/update_time", update_time, global_step)
            logger.add_scalar("time/rollout_time", rollout_time, global_step)
            logger.add_scalar(
                "time/rollout_fps",
                global_steps_per_iteration / rollout_time,
                global_step,
            )
            for k, v in cumulative_times.items():
                logger.add_scalar(f"time/total_{k}", v, global_step)
            logger.add_scalar(
                "time/total_rollout+update_time",
                cumulative_times["rollout_time"] + cumulative_times["update_time"],
                global_step,
            )
            if args.autotune:
                logger.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    if not args.evaluate and args.save_model:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        _save_dict = {
            "actor": actor.state_dict(),
            "qfs": [qf_target.state_dict() for qf_target in qf_targets],
            "log_alpha": log_alpha,
        }
        if args.obs_rms:
            _save_dict["obs_rms"] = obs_rms.state_dict()
        torch.save(
            _save_dict,
            model_path,
        )
        print(f"model saved to {model_path}")
        writer.close()
    envs.close()
