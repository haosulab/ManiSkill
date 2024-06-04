
from dataclasses import dataclass
import os
import random
import re
import shutil
import time
from typing import Optional

from tqdm import tqdm

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.trajectory import dataset
from mani_skill.utils import common, gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tyro

import mani_skill.envs
import h5py

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
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: str = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save the model checkpoints"""

    # Env specific arguments
    env_id: str = "PushCube-v1"
    """the environment id of the task"""
    num_envs: int = 16
    """the number of parallel environments"""
    num_eval_envs: int = 8
    """the number of parallel evaluation environments"""
    num_eval_steps: int = 50
    """the number of steps to take in evaluation environments"""
    log_freq: int = 1_000
    """logging frequency in terms of environment steps"""
    eval_freq: int = 100_000
    """evaluation frequency in terms of environment steps"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of environment steps"""

    # Algorithm specific arguments
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    buffer_size: int = 1_000_000
    """the replay memory buffer size"""
    buffer_device: str = "cpu"
    """where the replay buffer is stored. Can be 'cpu' or 'cuda' for GPU"""
    gamma: float = 0.9
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""
    learning_starts: int = 5_000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 1
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 1.0
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    training_freq: int = 64
    """training frequency (in steps)"""
    utd: float = 0.5
    """update to data ratio"""
    partial_reset: bool = False
    """whether to let parallel environments reset upon termination instead of truncation"""
    bootstrap_at_done: str = "always"
    """the bootstrap method to use when a done signal is received. Can be 'always' or 'never'"""

    # RFCL specific arguments
    dataset_path: str = ""
    """path to the trajectory.h5 file to use for RFCL"""
    num_demos: Optional[int] = None
    """number of demonstrations to load. If None all are loaded. If given a int, that many demos
    are sampled from the given dataset"""

    reverse_step_size: int = 4
    """the number of steps to reverse the curriculum by"""
    curriculum_method: str = "geometric"
    """the curriculum to use. Can be 'geometric' or 'uniform'"""
    # TODO not implemented
    per_demo_buffer_size: int = 3
    """number of sequential successes before considering advancing the curriculum """
    # TODO not implemented
    demo_horizon_to_max_steps_ratio: float = 3
    """the demo horizon to max steps ratio for dynamic timelimits for faster training with partial resets"""



    # to be filled in runtime
    grad_steps_per_iteration: int = 0
    """the number of gradient updates per iteration"""
    steps_per_env: int = 0
    """the number of steps each parallel env takes per iteration"""

@dataclass
class ReplayBufferSample:
    obs: torch.Tensor
    next_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
class ReplayBuffer:
    def __init__(self, env, num_envs: int, buffer_size: int, storage_device: torch.device, sample_device: torch.device):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.num_envs = num_envs
        self.storage_device = storage_device
        self.sample_device = sample_device
        self.obs = torch.zeros((buffer_size, num_envs) + env.single_observation_space.shape).to(storage_device)
        self.next_obs = torch.zeros((buffer_size, num_envs) + env.single_observation_space.shape).to(storage_device)
        self.actions = torch.zeros((buffer_size, num_envs) + env.single_action_space.shape).to(storage_device)
        self.logprobs = torch.zeros((buffer_size, num_envs)).to(storage_device)
        self.rewards = torch.zeros((buffer_size, num_envs)).to(storage_device)
        self.dones = torch.zeros((buffer_size, num_envs)).to(storage_device)
        self.values = torch.zeros((buffer_size, num_envs)).to(storage_device)

    def add(self, obs: torch.Tensor, next_obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, done: torch.Tensor):
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
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
    def sample(self, batch_size: int):
        if self.full:
            batch_inds = torch.randint(0, self.buffer_size, size=(batch_size, ))
        else:
            batch_inds = torch.randint(0, self.pos, size=(batch_size, ))
        env_inds = torch.randint(0, self.num_envs, size=(batch_size, ))
        return ReplayBufferSample(
            obs=self.obs[batch_inds, env_inds].to(self.sample_device),
            next_obs=self.next_obs[batch_inds, env_inds].to(self.sample_device),
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones=self.dones[batch_inds, env_inds].to(self.sample_device)
        )

# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        return self.net(x)


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        h, l = env.single_action_space.high, env.single_action_space.low
        self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))
        # will be saved in the state_dict

    def forward(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

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


class ReverseForwardCurriculumWrapper(gym.Wrapper):
    """Apply this before any auto reset wrapper"""
    def __init__(self, env, dataset_path,
                 reverse_curriculum_sampler: str = "uniform",
                 demo_horizon_to_max_steps_ratio: float = 3.0,
                 per_demo_buffer_size = 3,
                 reverse_step_size = 1,
                 traj_ids: list[str] = None,
                 eval_mode=False):
        super().__init__(env)
        self.curriculum_mode = "reverse" # "reverse" or "forward" or "none"
        """which curriculum to apply to modify env states during training"""
        self.eval_mode = eval_mode

        # Reverse curriculum specific configs
        self.reverse_curriculum_sampler = reverse_curriculum_sampler
        """choice of sampler"""
        self.demo_horizon_to_max_steps_ratio = demo_horizon_to_max_steps_ratio
        self.per_demo_buffer_size = per_demo_buffer_size
        self.reverse_step_size = reverse_step_size

        # Forward curriculum specific configs


        dataset_path = os.path.expanduser(dataset_path)
        h5py_file = h5py.File(dataset_path, "r")
        max_eps_len = -1
        env_states_flat_list = []
        if traj_ids is None:
            traj_ids = list(h5py_file.keys())
        traj_count = len(traj_ids)
        for traj_id in traj_ids:
            env_states = dataset.load_h5_data(h5py_file[traj_id]["env_states"])
            env_states_flat = common.flatten_state_dict(env_states)
            max_eps_len = max(len(env_states_flat), max_eps_len)
            env_states_flat_list.append(env_states_flat)
        self.env_states = torch.zeros((traj_count, max_eps_len, env_states_flat.shape[-1]), device=self.base_env.device)
        """environment states flattened into a matrix of shape (B, T, D) where B is the number of episodes, T is the maximum episode length, and D is the dimension of the state"""
        self.demo_curriculum_step = torch.zeros((traj_count,), dtype=torch.int32)
        """the current curriculum step/stage for each demonstration given. Used in reverse curricula options"""
        self.demo_horizon = torch.zeros((traj_count,), dtype=torch.int32, device=self.base_env.device)
        """length of each demo"""
        self.demo_solved = torch.zeros((traj_count,), dtype=torch.bool, device=self.base_env.device)
        """whether the demo is solved (meaning the curriculum stage has reached the end)"""


        for i, env_states_flat in enumerate(env_states_flat_list):
            self.env_states[i, :len(env_states_flat)] = torch.from_numpy(env_states_flat).float().to(self.base_env.device)
            self.demo_horizon[i] = len(env_states_flat)

        if not self.eval_mode:
            self.demo_curriculum_step = self.demo_horizon - 1
        h5py_file.close()

        self._demo_success_rate_buffer_pos = torch.zeros((traj_count, ), dtype=torch.int, device=self.base_env.device)
        self.demo_success_rate_buffers = torch.zeros((traj_count, self.per_demo_buffer_size), dtype=torch.bool, device=self.base_env.device)


        self.max_episode_steps = gym_utils.find_max_episode_steps_value(self.env)
        self.sampled_traj_indexes = torch.zeros((self.base_env.num_envs, ), dtype=torch.int, device=self.base_env.device)
        self.dynamic_max_episode_steps = torch.zeros((self.base_env.num_envs, ), dtype=torch.int, device=self.base_env.device)

        print(f"ReverseForwardCurriculumWrapper initialized. Loaded {traj_count} demonstrations. Trajectory IDs: {traj_ids} \n \
              Mean Length: {np.mean(self.demo_horizon.cpu().numpy())}, \
              Max Length: {np.max(self.demo_horizon.cpu().numpy())}")
    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.max_episode_steps is not None:
            if self.curriculum_mode == "reverse":
                truncated: torch.Tensor = (
                    self.base_env.elapsed_steps >= self.dynamic_max_episode_steps
                )
            else:
                # without dynamic time limits except all envs to be synced.
                # this might not be necessary though
                truncated: torch.Tensor = (
                    self.base_env.elapsed_steps >= self.max_episode_steps
                )
                assert truncated.any() == truncated.all()
        assert "success" in info, "Reverse curriculum wrapper currently requires there to be a success key in the info dict"

        if not self.eval_mode:
            if self.curriculum_mode == "reverse" and truncated.any():
                truncated_traj_idxs = self.sampled_traj_indexes[truncated]
                self.demo_success_rate_buffers[truncated_traj_idxs, self._demo_success_rate_buffer_pos[truncated_traj_idxs]] = info["success"]

                # advance curriculum. code below is indexing arrays shaped by the number of demos
                self._demo_success_rate_buffer_pos[truncated_traj_idxs] = (self._demo_success_rate_buffer_pos[truncated_traj_idxs] + 1) % self.per_demo_buffer_size
                per_demo_success_rates = self.demo_success_rate_buffers.float().mean(dim=1)
                can_advance = per_demo_success_rates > 0.9
                self.demo_curriculum_step[can_advance] -= self.reverse_step_size
                self.demo_success_rate_buffers[can_advance, :] = 0
                self.demo_solved[self.demo_curriculum_step < 0] = True
                self.demo_curriculum_step = torch.clamp(self.demo_curriculum_step, 0)
            elif self.curriculum_mode == "forward":
                pass

        return obs, reward, terminated, truncated, info
    def reset(self, *, seed=None, options=dict()):
        super().reset(seed=seed, options=options)
        if "env_idx" in options:
            env_idx = options["env_idx"]
        else:
            env_idx = torch.arange(0, self.base_env.num_envs, device=self.base_env.device)
        if self.curriculum_mode == "reverse":
            # set initial state accordingly
            b = self.base_env.num_envs
            # TODO (stao): handle partial resets later
            self.sampled_traj_indexes[env_idx] = torch.from_numpy(self.base_env._episode_rng.randint(0, len(self.env_states), size=(b, ))).int().to(self.base_env.device)
            if self.eval_mode:
                self.base_env.set_state(self.env_states[self.sampled_traj_indexes, 5 + torch.zeros((b, ), dtype=torch.int, device=self.base_env.device)])
            elif self.reverse_curriculum_sampler == "geometric":
                x_start_steps_density_list = [0.5, 0.25, 0.125, 0.125 / 2, 0.125 / 2]
                sampled_offsets = torch.from_numpy(self.base_env._episode_rng.randint(0, len(x_start_steps_density_list), size=(b, ))).to(self.base_env.device)
                x_start_steps = self.demo_curriculum_step[self.sampled_traj_indexes] + sampled_offsets
                x_start_steps = torch.clamp(x_start_steps, torch.zeros((b, ), device=self.base_env.device), self.demo_horizon[self.sampled_traj_indexes] - 1).int()
            self.dynamic_max_episode_steps[env_idx] = 8 + (self.demo_horizon[self.sampled_traj_indexes] - x_start_steps) // self.demo_horizon_to_max_steps_ratio
            self.base_env.set_state(self.env_states[self.sampled_traj_indexes, x_start_steps])

        obs = self.base_env.get_obs()
        return obs, {}


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.grad_steps_per_iteration = int(args.training_freq * args.utd)
    args.steps_per_env = args.training_freq // args.num_envs
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    writer = None
    if not args.evaluate:
        print("Running training")
        if os.path.exists(f"runs/{run_name}"):
            shutil.rmtree(f"runs/{run_name}")
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
    else:
        print("Running evaluation")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_kwargs = dict(obs_mode="state", control_mode="pd_joint_delta_pos", render_mode="rgb_array", reward_mode="sparse", sim_backend="gpu")
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // 50) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=50, video_fps=30)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.evaluate, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)

    traj_ids = None
    h5py_file = h5py.File(os.path.expanduser(args.dataset_path), "r")
    traj_ids = list(h5py_file.keys())
    h5py_file.close()

    if args.num_demos is not None:
        traj_ids = np.random.choice(traj_ids, size=args.num_demos, replace=False)

    curriculum_wrapped_envs = ReverseForwardCurriculumWrapper(
        envs, args.dataset_path,
        reverse_curriculum_sampler=args.curriculum_method,
        demo_horizon_to_max_steps_ratio=args.demo_horizon_to_max_steps_ratio,
        per_demo_buffer_size=args.per_demo_buffer_size,
        reverse_step_size=args.reverse_step_size,
        traj_ids=traj_ids,
        eval_mode=False
    )
    # eval_envs = ReverseCurriculumWrapper(
    #     eval_envs, args.dataset_path,
    #     curriculum=args.curriculum_method,
    #     demo_horizon_to_max_steps_ratio=args.demo_horizon_to_max_steps_ratio,
    #     per_demo_buffer_size=args.per_demo_buffer_size,
    #     reverse_step_size=args.reverse_step_size,
    #     traj_ids=traj_ids,
    #     eval_mode=True
    # )
    envs = ManiSkillVectorEnv(curriculum_wrapped_envs, args.num_envs, ignore_terminations=not args.partial_reset, **env_kwargs)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.partial_reset, **env_kwargs)

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"


    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint)
        actor.load_state_dict(ckpt['actor'])
        qf1.load_state_dict(ckpt['qf1'])
        qf2.load_state_dict(ckpt['qf2'])
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        env=envs,
        num_envs=args.num_envs,
        buffer_size=args.buffer_size,
        storage_device=torch.device(args.buffer_device),
        sample_device=device
    )
    h5py_file = h5py.File(os.path.expanduser(args.dataset_path), "r")
    total_offline_transitions = 0
    for traj_id in traj_ids:
        trajectory = common.to_cpu_tensor(dataset.load_h5_data(h5py_file[traj_id]))
        total_offline_transitions += len(trajectory["obs"]) - 1

    offline_rb = ReplayBuffer(
        env=envs,
        num_envs=1,
        buffer_size=total_offline_transitions,
        storage_device=torch.device(args.buffer_device),
        sample_device=device
    )
    # fill offline replay buffer
    h5py_file = h5py.File(os.path.expanduser(args.dataset_path), "r")
    for traj_id in h5py_file.keys():
        trajectory = common.to_cpu_tensor(dataset.load_h5_data(h5py_file[traj_id]))
        for i in range(len(trajectory["obs"]) - 1):
            done = False
            reward = 0.0
            if args.bootstrap_at_done == 'always':
                if trajectory["success"][i]:
                    reward = 1.0
            else:
                raise ValueError("Cannot run RFCL in SAC with bootstrap_at_done not equal to 'always'")
            offline_rb.add(
                obs=trajectory["obs"][i][0],
                next_obs=trajectory["obs"][i + 1][0],
                action=trajectory["actions"][i],
                reward=torch.tensor(reward),
                done=torch.tensor(done)
            )
    h5py_file.close()
    # TRY NOT TO MODIFY: start the game
    obs, info = envs.reset(seed=args.seed) # in Gymnasium, seed is given to reset() instead of seed()
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    global_step = 0
    global_update = 0
    learning_has_started = False

    global_steps_per_iteration = args.num_envs * (args.steps_per_env)
    pbar = tqdm(total=args.total_timesteps, initial=0, position=0, desc=run_name)
    while global_step < args.total_timesteps:
        if args.eval_freq > 0 and (global_step - args.training_freq) // args.eval_freq < global_step // args.eval_freq:
            # evaluate
            actor.eval()
            print("Evaluating")
            eval_envs.reset()
            returns = []
            eps_lens = []
            successes = []
            failures = []
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_obs, _, eval_terminations, eval_truncations, eval_infos = eval_envs.step(actor.get_eval_action(eval_obs))
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        eps_lens.append(eval_infos["final_info"]["elapsed_steps"][mask].cpu().numpy())
                        returns.append(eval_infos["final_info"]["episode"]["r"][mask].cpu().numpy())
                        if "success" in eval_infos:
                            successes.append(eval_infos["final_info"]["success"][mask].cpu().numpy())
                        if "fail" in eval_infos:
                            failures.append(eval_infos["final_info"]["fail"][mask].cpu().numpy())
            returns = np.concatenate(returns)
            eps_lens = np.concatenate(eps_lens)
            print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {len(eps_lens)} episodes")
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
            actor.train()
            if args.evaluate:
                break

            if args.save_model:
                model_path = f"runs/{run_name}/ckpt_{global_step}.pt"
                torch.save({
                    'actor': actor.state_dict(),
                    'qf1': qf1_target.state_dict(),
                    'qf2': qf2_target.state_dict(),
                    'log_alpha': log_alpha,
                }, model_path)
                print(f"model saved to {model_path}")


        solved_frac = (curriculum_wrapped_envs.demo_solved).float().mean().item()
        # handle stage 1 to stage 2 training transition
        if solved_frac >= 0.9:
            print("Reverse solved >= 0.9 of demos. Stopping stage 1 training and beginning stage 2")
            writer.add_scalar("charts/stage_1_steps", global_step, global_step)
            envs.curriculum_mode = "none"
            # reset the environment and begin training as if training anew
            envs.reset()
            print(f"Loading current online replay buffer as offline replay buffer and resetting online buffer")
            offline_rb = rb
            rb = ReplayBuffer(
                env=envs,
                num_envs=args.num_envs,
                buffer_size=args.buffer_size,
                storage_device=torch.device(args.buffer_device),
                sample_device=device
            )





        # Collect samples from environemnts
        rollout_time = time.time()
        for local_step in range(args.steps_per_env):
            global_step += 1 * args.num_envs

            # ALGO LOGIC: put action logic here
            if not learning_has_started:
                actions = torch.tensor(envs.action_space.sample(), dtype=torch.float32, device=device)
            else:
                actions, _, _ = actor.get_action(obs)
                actions = actions.detach()
                # actions = actions.detach().cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            real_next_obs = next_obs.clone()
            if args.bootstrap_at_done == 'always':
                next_done = torch.zeros_like(terminations).to(torch.float32)
            else:
                next_done = (terminations | truncations).to(torch.float32)
            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                real_next_obs[done_mask] = infos["final_observation"][done_mask]
                episodic_return = final_info['episode']['r'][done_mask].cpu().numpy().mean()
                if "success" in final_info:
                    writer.add_scalar("charts/success_rate", final_info["success"][done_mask].cpu().numpy().mean(), global_step)
                if "fail" in final_info:
                    writer.add_scalar("charts/fail_rate", final_info["fail"][done_mask].cpu().numpy().mean(), global_step)
                writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                writer.add_scalar("charts/episodic_length", final_info["elapsed_steps"][done_mask].cpu().numpy().mean(), global_step)

            rb.add(obs, real_next_obs, actions, rewards, next_done)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
        rollout_time = time.time() - rollout_time
        pbar.update(global_steps_per_iteration)
        # ALGO LOGIC: training.
        if global_step < args.learning_starts:
            continue

        update_time = time.time()
        learning_has_started = True
        for local_update in range(args.grad_steps_per_iteration):
            global_update += 1
            data = rb.sample(args.batch_size // 2)
            offline_data = offline_rb.sample(args.batch_size // 2)
            data.obs = torch.cat([data.obs, offline_data.obs], dim=0)
            data.next_obs = torch.cat([data.next_obs, offline_data.next_obs], dim=0)
            data.actions = torch.cat([data.actions, offline_data.actions], dim=0)
            data.rewards = torch.cat([data.rewards, offline_data.rewards], dim=0)
            data.dones = torch.cat([data.dones, offline_data.dones], dim=0)

            # update the value networks
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_obs)
                qf1_next_target = qf1_target(data.next_obs, next_state_actions)
                qf2_next_target = qf2_target(data.next_obs, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)
                # data.dones is "stop_bootstrap", which is computed earlier according to args.bootstrap_at_done

            qf1_a_values = qf1(data.obs, data.actions).view(-1)
            qf2_a_values = qf2(data.obs, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # update the policy network
            if global_update % args.policy_frequency == 0:  # TD 3 Delayed update support
                pi, log_pi, _ = actor.get_action(data.obs)
                qf1_pi = qf1(data.obs, pi)
                qf2_pi = qf2(data.obs, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

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
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
        update_time = time.time() - update_time

        # Log training-related data
        if (global_step - args.training_freq) // args.log_freq < global_step // args.log_freq:
            writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
            writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
            writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
            writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
            writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            writer.add_scalar("losses/alpha", alpha, global_step)
            writer.add_scalar("charts/update_time", update_time, global_step)
            writer.add_scalar("charts/rollout_time", rollout_time, global_step)
            writer.add_scalar("charts/rollout_fps", global_steps_per_iteration / rollout_time, global_step)
            start_step_fracs = torch.divide(curriculum_wrapped_envs.demo_curriculum_step, curriculum_wrapped_envs.demo_horizon - 1).cpu().numpy()
            writer.add_histogram("charts/start_step_frac_dist", start_step_fracs, global_step),
            writer.add_scalar("charts/start_step_frac_avg", start_step_fracs.mean(), global_step)
            if args.autotune:
                writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    if not args.evaluate and args.save_model:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save({
            'actor': actor.state_dict(),
            'qf1': qf1_target.state_dict(),
            'qf2': qf2_target.state_dict(),
            'log_alpha': log_alpha,
        }, model_path)
        print(f"model saved to {model_path}")
        writer.close()
    envs.close()
