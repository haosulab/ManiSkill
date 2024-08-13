ALGO_NAME = 'BC_Diffusion_state_UNet'

import argparse
import os
import random
from distutils.util import strtobool
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from diffusion_policy.evaluate import evaluate
from mani_skill.utils.registration import REGISTERED_ENVS

from collections import defaultdict

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.dataloader import DataLoader
from diffusion_policy.utils import IterationBasedBatchSampler, worker_init_fn
from diffusion_policy.make_env import make_env
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from dataclasses import dataclass, field
from typing import Optional, List
import tyro
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
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    env_id: str = "PegInsertionSide-v0"
    """the id of the environment"""
    demo_path: str = 'data/ms2_official_demos/rigid_body/PegInsertionSide-v0/trajectory.state.pd_ee_delta_pose.h5'
    """the path of demo dataset (pkl or h5)"""
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    total_iters: int = 1_000_000
    """total timesteps of the experiments"""
    batch_size: int = 1024
    """the batch size of sample from the replay memory"""

    # Diffusion Policy specific arguments
    lr: float = 1e-4
    obs_horizon: int = 2 # Seems not very important in ManiSkill, 1, 2, 4 work well
    act_horizon: int = 8 # Seems not very important in ManiSkill, 4, 8, 15 work well
    pred_horizon: int = 16 # 16->8 leads to worse performance, maybe it is like generate a half image; 16->32, improvement is very marginal
    diffusion_step_embed_dim: int = 64 # not very important
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256]) # default setting is about ~4.5M params
    n_groups: int = 8 # jigu says it is better to let each group have at least 8 channels; it seems 4 and 8 are simila
    max_episode_steps: Optional[int] = None
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""

    # Environment/experiment specific arguments
    output_dir: str = 'output'
    log_freq: int = 1000
    eval_freq: int = 5000
    save_freq: Optional[int] = None
    num_eval_episodes: int = 100
    num_eval_envs: int = 10
    sim_backend: str = "cpu"
    num_dataload_workers: int = 0
    control_mode: str = 'pd_joint_delta_pos'
    obj_ids: List[str] = field(default_factory=list)

    # additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms
    demo_type: Optional[str] = None


class SmallDemoDataset_DiffusionPolicy(Dataset): # Load everything into GPU memory
    def __init__(self, data_path, device, num_traj):
        if data_path[-4:] == '.pkl':
            raise NotImplementedError()
        else:
            from diffusion_policy.utils import load_demo_dataset
            trajectories = load_demo_dataset(data_path, num_traj=num_traj, concat=False)
            # trajectories['observations'] is a list of np.ndarray (L+1, obs_dim)
            # trajectories['actions'] is a list of np.ndarray (L, act_dim)

        for k, v in trajectories.items():
            for i in range(len(v)):
                trajectories[k][i] = torch.Tensor(v[i]).to(device)

        # Pre-compute all possible (traj_idx, start, end) tuples, this is very specific to Diffusion Policy
        if 'delta_pos' in args.control_mode or args.control_mode == 'base_pd_joint_vel_arm_pd_joint_vel':
            self.pad_action_arm = torch.zeros((trajectories['actions'][0].shape[1]-1,), device=device)
            # to make the arm stay still, we pad the action with 0 in 'delta_pos' control mode
            # gripper action needs to be copied from the last action
        # else:
        #     raise NotImplementedError(f'Control Mode {args.control_mode} not supported')
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon = args.obs_horizon, args.pred_horizon
        self.slices = []
        num_traj = len(trajectories['actions'])
        total_transitions = 0
        for traj_idx in range(num_traj):
            L = trajectories['actions'][traj_idx].shape[0]
            assert trajectories['observations'][traj_idx].shape[0] == L + 1
            total_transitions += L

            # |o|o|                             observations: 2
            # | |a|a|a|a|a|a|a|a|               actions executed: 8
            # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
            pad_before = obs_horizon - 1
            # Pad before the trajectory, so the first action of an episode is in "actions executed"
            # obs_horizon - 1 is the number of "not used actions"
            pad_after = pred_horizon - obs_horizon
            # Pad after the trajectory, so all the observations are utilized in training
            # Note that in the original code, pad_after = act_horizon - 1, but I think this is not the best choice
            self.slices += [
                (traj_idx, start, start + pred_horizon) for start in range(-pad_before, L - pred_horizon + pad_after)
            ]  # slice indices follow convention [start, end)

        print(f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}")

        self.trajectories = trajectories

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories['actions'][traj_idx].shape

        obs_seq = self.trajectories['observations'][traj_idx][max(0, start):start+self.obs_horizon]
        # start+self.obs_horizon is at least 1
        act_seq = self.trajectories['actions'][traj_idx][max(0, start):end]
        if start < 0: # pad before the trajectory
            obs_seq = torch.cat([obs_seq[0].repeat(-start, 1), obs_seq], dim=0)
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L: # pad after the trajectory
            gripper_action = act_seq[-1, -1]
            pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            act_seq = torch.cat([act_seq, pad_action.repeat(end-L, 1)], dim=0)
            # making the robot (arm and gripper) stay still
        assert obs_seq.shape[0] == self.obs_horizon and act_seq.shape[0] == self.pred_horizon
        return {
            'observations': obs_seq,
            'actions': act_seq,
        }

    def __len__(self):
        return len(self.slices)


class Agent(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        assert len(env.single_observation_space.shape) == 2 # (obs_horizon, obs_dim)
        assert len(env.single_action_space.shape) == 1 # (act_dim, )
        assert (env.single_action_space.high == 1).all() and (env.single_action_space.low == -1).all()
        # denoising results will be clipped to [-1,1], so the action should be in [-1,1] as well
        self.act_dim = env.single_action_space.shape[0]

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.act_dim, # act_horizon is not used (U-Net doesn't care)
            global_cond_dim=np.prod(env.single_observation_space.shape), # obs_horizon * obs_dim
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2', # has big impact on performance, try not to change
            clip_sample=True, # clip output to [-1,1] to improve stability
            prediction_type='epsilon' # predict noise (instead of denoised action)
        )

        self.get_eval_action = self.get_action

    # def forward(self, obs_seq):
    #     raise NotImplementedError()

    def compute_loss(self, obs_seq, action_seq):
        B = obs_seq.shape[0]

        # observation as FiLM conditioning
        obs_cond = obs_seq.flatten(start_dim=1) # (B, obs_horizon * obs_dim)

        # sample noise to add to actions
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=device)

        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()

        # add noise to the clean images(actions) according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        noisy_action_seq = self.noise_scheduler.add_noise(
            action_seq, noise, timesteps)

        # predict the noise residual
        noise_pred = self.noise_pred_net(
            noisy_action_seq, timesteps, global_cond=obs_cond)

        return F.mse_loss(noise_pred, noise)

    def get_action(self, obs_seq):
        # init scheduler
        # self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        # set_timesteps will change noise_scheduler.timesteps is only used in noise_scheduler.step()
        # noise_scheduler.step() is only called during inference
        # if we use DDPM, and inference_diffusion_steps == train_diffusion_steps, then we can skip this

        # obs_seq: (B, obs_horizon, obs_dim)
        B = obs_seq.shape[0]
        with torch.no_grad():
            obs_cond = obs_seq.flatten(start_dim=1) # (B, obs_horizon * obs_dim)

            # initialize action from Guassian noise
            noisy_action_seq = torch.randn((B, self.pred_horizon, self.act_dim), device=obs_seq.device)

            for k in self.noise_scheduler.timesteps:
                # predict noise
                noise_pred = self.noise_pred_net(
                    sample=noisy_action_seq,
                    timestep=k,
                    global_cond=obs_cond,
                )

                # inverse diffusion step (remove noise)
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=noisy_action_seq,
                ).prev_sample

        # only take act_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end] # (B, act_horizon, act_dim)

def save_ckpt(run_name, tag):
    os.makedirs(f'runs/{run_name}/checkpoints', exist_ok=True)
    ema.copy_to(ema_agent.parameters())
    torch.save({
        'agent': agent.state_dict(),
        'ema_agent': ema_agent.state_dict(),
    }, f'runs/{run_name}/checkpoints/{tag}.pt')

if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    if args.demo_path.endswith('.h5'):
        import json
        json_file = args.demo_path[:-2] + 'json'
        with open(json_file, 'r') as f:
            demo_info = json.load(f)
            if 'control_mode' in demo_info['env_info']['env_kwargs']:
                control_mode = demo_info['env_info']['env_kwargs']['control_mode']
            elif 'control_mode' in demo_info['episodes'][0]:
                control_mode = demo_info['episodes'][0]['control_mode']
            else:
                raise Exception('Control mode not found in json')
            assert control_mode == args.control_mode, f"Control mode mismatched. Dataset has control mode {control_mode}, but args has control mode {args.control_mode}"
    assert args.obs_horizon + args.act_horizon - 1 <= args.pred_horizon
    assert args.obs_horizon >= 1 and args.act_horizon >= 1 and args.pred_horizon >= 1

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    max_episode_steps = args.max_episode_steps if args.max_episode_steps is not None else REGISTERED_ENVS[args.env_id].max_episode_steps
    # note we set reconfiguration_freq to 1 to ensure that the env is reconfigured after each episode to sufficiently randomize object geometries
    env_kwargs = dict(control_mode=args.control_mode, reward_mode="sparse", obs_mode="state", render_mode="rgb_array", max_episode_steps=max_episode_steps, reconfiguration_freq=1)
    other_kwargs = dict(obs_horizon=args.obs_horizon)
    eval_envs = make_env(args.env_id, args.num_eval_envs, args.sim_backend, args.seed, env_kwargs, other_kwargs, video_dir=f'runs/{run_name}/videos' if args.capture_video else None)
    eval_envs.reset(seed=args.seed) # seed eval_envs here
    envs = eval_envs
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # dataloader setup
    dataset = SmallDemoDataset_DiffusionPolicy(args.demo_path, device, num_traj=args.num_demos)
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
    )
    if args.num_demos is None:
        args.num_demos = len(dataset)

    # agent setup
    agent = Agent(envs, args).to(device)
    optimizer = optim.AdamW(params=agent.parameters(),
        lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_iters,
    )

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = Agent(envs, args).to(device)



    if args.track:
        import wandb
        config = vars(args)
        config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, env_horizon=max_episode_steps)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            group="DiffusionPolicy",
            tags=["diffusion_policy"]
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    import json
    with open(f'runs/{run_name}/args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    # ---------------------------------------------------------------------------- #
    # Training begins.
    # ---------------------------------------------------------------------------- #
    agent.train()
    best_success_once_rate = -1
    best_success_at_end_rate = -1

    timings = defaultdict(float)

    for iteration, data_batch in enumerate(train_dataloader):
        cur_iter = iteration + 1

        # # copy data from cpu to gpu
        # data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items()}

        # forward and compute loss
        total_loss = agent.compute_loss(
            obs_seq=data_batch['observations'], # (B, L, obs_dim)
            action_seq=data_batch['actions'], # (B, L, act_dim)
        )

        # backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step() # step lr scheduler every batch, this is different from standard pytorch behavior
        last_tick = time.time()

        # update Exponential Moving Average of the model weights
        ema.step(agent.parameters())
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if cur_iter % args.log_freq == 0:
            print(f"Iteration {cur_iter}, loss: {total_loss.item()}")
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], cur_iter)
            writer.add_scalar("losses/total_loss", total_loss.item(), cur_iter)
            for k, v in timings.items():
                writer.add_scalar(f"time/{k}", v, cur_iter)
        # Evaluation
        if cur_iter % args.eval_freq == 0:
            print('Evaluating')
            last_tick = time.time()

            ema.copy_to(ema_agent.parameters())
            result = evaluate(args.num_eval_episodes, ema_agent, eval_envs, device)
            timings["eval"] += time.time() - last_tick

            for k, v in result.items():
                writer.add_scalar(f"eval/{k}", np.mean(v), cur_iter)
            sr_once = np.mean(result['success_once'])
            sr_at_end = np.mean(result['success_at_end'])
            print(f"Evaluated {len(result['return'])} episodes")
            print(f"Success Once Rate: {sr_once:.4f}")
            print(f"Success At End Rate: {sr_at_end:.4f}")
            if sr_once > best_success_once_rate:
                best_success_once_rate = sr_once
                save_ckpt(run_name, 'best_eval_success_rate')
                print(f'New best success rate: {sr_once:.4f}. Saving checkpoint.')
            if sr_at_end > best_success_at_end_rate:
                best_success_at_end_rate = sr_at_end
                save_ckpt(run_name, 'best_eval_success_rate_at_end')
                print(f'New best success rate at end: {sr_at_end:.4f}. Saving checkpoint.')
        # Checkpoint
        if args.save_freq and cur_iter % args.save_freq == 0:
            save_ckpt(run_name, str(cur_iter))
    envs.close()
    writer.close()
