ALGO_NAME = 'BC_ACT_rgb'

import argparse
import os
import random
from distutils.util import strtobool
from functools import partial
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from act.evaluate import evaluate
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils
from mani_skill.utils.registration import REGISTERED_ENVS

from collections import defaultdict

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.dataloader import DataLoader
from act.utils import IterationBasedBatchSampler, worker_init_fn
from act.make_env import make_eval_envs
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from act.detr.backbone import build_backbone
from act.detr.transformer import build_transformer
from act.detr.detr_vae import build_encoder, DETRVAE
from dataclasses import dataclass, field
from typing import Optional, List, Dict
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

    env_id: str = "PickCube-v1"
    """the id of the environment"""
    demo_path: str = 'pickcube.trajectory.rgbd.pd_joint_delta_pos.cpu.h5'
    """the path of demo dataset (pkl or h5)"""
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    total_iters: int = 1_000_000
    """total timesteps of the experiment"""
    batch_size: int = 128
    """the batch size of sample from the replay memory"""

    # ACT specific arguments
    lr: float = 1e-4
    """the learning rate of the Action Chunking with Transformers"""
    kl_weight: float = 10 
    temporal_agg: bool = True

    # Backbone
    position_embedding: str = 'sine'
    backbone: str = 'resnet18'
    lr_backbone: float = 1e-5
    masks: bool = False
    dilation: bool = False

    # Transformer
    enc_layers: int = 4
    dec_layers: int = 7
    dim_feedforward: int = 1600
    hidden_dim: int = 512
    dropout: float = 0.1
    nheads: int = 8
    num_queries: int = 30
    pre_norm: bool = False

    # Environment/experiment specific arguments
    max_episode_steps: Optional[int] = None
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    log_freq: int = 1000
    """the frequency of logging the training metrics"""
    eval_freq: int = 5000
    """the frequency of evaluating the agent on the evaluation environments"""
    save_freq: Optional[int] = None
    """the frequency of saving the model checkpoints. By default this is None and will only save checkpoints based on the best evaluation metrics."""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 10
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data in the torch dataloader"""
    control_mode: str = 'pd_joint_delta_pos'
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""

    # additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms
    demo_type: Optional[str] = None


class FlattenRGBDObservationWrapper(gym.ObservationWrapper):
    """
    Flattens the rgbd mode observations into a dictionary with two keys, "rgbd" and "state"

    Args:
        rgb (bool): Whether to include rgb images in the observation
        depth (bool): Whether to include depth images in the observation
        state (bool): Whether to include state data in the observation

    Note that the returned observations will have a "rgbd" or "rgb" or "depth" key depending on the rgb/depth bool flags.
    """

    def __init__(self, env, rgb=True, depth=True, state=True) -> None:
        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        self.include_rgb = rgb
        self.include_depth = depth
        self.include_state = state
        self.transforms = T.Compose(
            [
                T.Resize((224, 224), antialias=True),
            ]
        )  # resize the input image to be at least 224x224
        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: Dict):
        sensor_data = observation.pop("sensor_data")
        del observation["sensor_param"]
        images = []
        for cam_data in sensor_data.values():
            if self.include_rgb:
                resized_rgb = self.transforms(
                    cam_data["rgb"].permute(0, 3, 1, 2) 
                )  # (1, 3, 224, 224)
                images.append(resized_rgb)
            if self.include_depth:
                resized_depth = self.transforms(
                    cam_data["depth"].permute(0, 3, 1, 2) 
                )  # (1, 1, 224, 224)
                images.append(resized_depth)

        images = torch.stack(images, dim=1) # (1, num_cams, C, 224, 224)

        # flatten the rest of the data which should just be state data
        observation = common.flatten_state_dict(observation, use_torch=True)
        ret = dict()
        if self.include_state:
            ret["state"] = observation
        if self.include_rgb and not self.include_depth:
            ret["rgb"] = images
        elif self.include_rgb and self.include_depth:
            ret["rgbd"] = images
        elif self.include_depth and not self.include_rgb:
            ret["depth"] = images
        return ret


class SmallDemoDataset_ACTPolicy(Dataset): # Load everything into memory
    def __init__(self, data_path, num_queries, num_traj, include_depth=False):
        if data_path[-4:] == '.pkl':
            raise NotImplementedError()
        else:
            from act.utils import load_demo_dataset
            trajectories = load_demo_dataset(data_path, num_traj=num_traj, concat=False)
            # trajectories['observations'] is a list of np.ndarray (L+1, obs_dim)
            # trajectories['actions'] is a list of np.ndarray (L, act_dim)
        print('Raw trajectory loaded, start to pre-process the observations...')

        self.include_depth = include_depth
        self.transforms = T.Compose(
            [
                T.Resize((224, 224), antialias=True),
            ]
        )  # pre-trained models from torchvision.models expect input image to be at least 224x224

        # Pre-process the observations, make them align with the obs returned by the FlattenRGBDObservationWrapper
        obs_traj_dict_list = []
        for obs_traj_dict in trajectories['observations']:
            obs_traj_dict = self.process_obs(obs_traj_dict)
            obs_traj_dict_list.append(obs_traj_dict)
        trajectories['observations'] = obs_traj_dict_list
        self.obs_keys = list(obs_traj_dict.keys())

        # Pre-process the actions
        for i in range(len(trajectories['actions'])):
            trajectories['actions'][i] = torch.Tensor(trajectories['actions'][i])
        print('Obs/action pre-processing is done.')

        # When the robot reaches the goal state, its joints and gripper fingers need to remain stationary
        if 'delta_pos' in args.control_mode or args.control_mode == 'base_pd_joint_vel_arm_pd_joint_vel':
            self.pad_action_arm = torch.zeros((trajectories['actions'][0].shape[1]-1,))
            # to make the arm stay still, we pad the action with 0 in 'delta_pos' control mode
            # gripper action needs to be copied from the last action
        # else:
        #     raise NotImplementedError(f'Control Mode {args.control_mode} not supported')

        self.num_queries = num_queries
        self.trajectories = trajectories
        self.norm_stats = self.get_norm_stats()

    def __getitem__(self, index):
        traj_idx = index
        episode_len = self.trajectories['actions'][traj_idx].shape[0]
        start_ts = np.random.choice(episode_len)

        # get state at start_ts only
        state = self.trajectories['observations'][traj_idx]['state'][start_ts]

        # get num_queries actions starting from start_ts
        act_seq = self.trajectories['actions'][traj_idx][start_ts:start_ts+self.num_queries]
        action_len = act_seq.shape[0]

        # normalize state and act_seq
        state = (state - self.norm_stats["state_mean"]) / self.norm_stats["state_std"]
        act_seq = (act_seq - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

        # Pad after the trajectory, so all the observations are utilized in training
        if action_len < self.num_queries:
            gripper_action = act_seq[-1, -1]
            pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            act_seq = torch.cat([act_seq, pad_action.repeat(self.num_queries-action_len, 1)], dim=0)
            # making the robot (arm and gripper) stay still

        # get rgb or rgbd data at start_ts and combine with state to form obs
        if self.include_depth:
            rgbd = self.trajectories['observations'][traj_idx]['rgbd'][start_ts]
            obs = dict(state=state, rgbd=rgbd)
        else:
            rgb = self.trajectories['observations'][traj_idx]['rgb'][start_ts]
            obs = dict(state=state, rgb=rgb)

        return {
            'observations': obs,
            'actions': act_seq,
        }

    def __len__(self):
        return len(self.trajectories['actions'])
    
    def process_obs(self, obs_dict):
        # get rgbd data
        sensor_data = obs_dict.pop("sensor_data")
        del obs_dict["sensor_param"]
        images = []
        for cam_data in sensor_data.values():
            cam_image = []
            cam_image.append(torch.from_numpy(cam_data["rgb"]))
            if self.include_depth:
                cam_image.append(torch.from_numpy(cam_data["depth"]))
            cam_image = torch.concat(cam_image, axis=-1) # (ep_len, H, W, C)
            resized_cam_image = self.transforms(
                cam_image.permute(0, 3, 1, 2) 
            )  # (ep_len, C, 224, 224); pre-trained models from torchvision.models expect input image to be at least 224x224
            images.append(resized_cam_image)
        images = torch.stack(images, dim=1) # (ep_len, num_cams, C, 224, 224)

        # flatten the rest of the data which should just be state data
        obs_dict['extra'] = {k: v[:, None] if len(v.shape) == 1 else v for k, v in obs_dict['extra'].items()} # dirty fix for data that has one dimension (e.g. is_grasped)
        obs_dict = common.flatten_state_dict(obs_dict, use_torch=True)

        processed_obs = dict(state=obs_dict, rgbd=images) if self.include_depth else dict(state=obs_dict, rgb=images)

        return processed_obs

    def get_norm_stats(self):
        all_state_data = []
        all_action_data = []
        for traj_idx in range(len(self)):
            state = self.trajectories['observations'][traj_idx]['state']
            action_seq = self.trajectories['actions'][traj_idx]
            all_state_data.append(state)
            all_action_data.append(action_seq)
        all_state_data = torch.concatenate(all_state_data)
        all_action_data = torch.concatenate(all_action_data)

        # normalize state data
        state_mean = all_state_data.mean(dim=0)
        state_std = all_state_data.std(dim=0)
        state_std = torch.clip(state_std, 1e-2, np.inf) # clipping

        # normalize action data
        action_mean = all_action_data.mean(dim=0, keepdim=True)
        action_std = all_action_data.std(dim=0, keepdim=True)
        action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

        # store example rgb (or rgbd) data
        if self.include_depth:
            visual_obs = self.trajectories['observations'][traj_idx]['rgbd'][0]
        else:
            visual_obs = self.trajectories['observations'][traj_idx]['rgb'][0]

        stats = {"action_mean": action_mean, "action_std": action_std,
                 "state_mean": state_mean, "state_std": state_std, 
                 "example_state": state, "example_visual_obs": visual_obs}

        return stats
    

class Agent(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        assert len(env.single_observation_space['state'].shape) == 1 # (obs_dim,)
        assert len(env.single_observation_space['rgb'].shape) == 4 # (num_cams, C, H, W)
        assert len(env.single_action_space.shape) == 1 # (act_dim,)
        assert (env.single_action_space.high == 1).all() and (env.single_action_space.low == -1).all()

        self.state_dim = env.single_observation_space['state'].shape[0]
        self.act_dim = env.single_action_space.shape[0]
        self.kl_weight = args.kl_weight
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        # CNN backbone
        backbones = []
        backbone = build_backbone(args)
        backbones.append(backbone)

        # CVAE decoder
        transformer = build_transformer(args)

        # CVAE encoder
        encoder = build_encoder(args)

        # ACT ( CVAE encoder + (CNN backbones + CVAE decoder) )
        self.model = DETRVAE(
            backbones,
            transformer,
            encoder,
            state_dim=self.state_dim,
            action_dim=self.act_dim,
            num_queries=args.num_queries,
        )

    def compute_loss(self, obs, action_seq):
        # normalize rgb data
        obs['rgb'] = obs['rgb'].float() / 255.0
        obs['rgb'] = self.normalize(obs['rgb'])

        # forward pass
        a_hat, (mu, logvar) = self.model(obs, action_seq)

        # compute l1 loss and kl loss
        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
        all_l1 = F.l1_loss(action_seq, a_hat, reduction='none')
        l1 = all_l1.mean()

        # store all loss
        loss_dict = dict()
        loss_dict['l1'] = l1
        loss_dict['kl'] = total_kld[0]
        loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
        return loss_dict

    def get_action(self, obs):
        # normalize rgb data
        obs['rgb'] = obs['rgb'].float() / 255.0
        obs['rgb'] = self.normalize(obs['rgb'])

        # forward pass
        a_hat, (_, _) = self.model(obs) # no action, sample from prior

        return a_hat


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_kwargs = dict(control_mode=args.control_mode, reward_mode="sparse", obs_mode="rgb", render_mode="rgb_array")
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    other_kwargs = None
    wrappers = [partial(FlattenRGBDObservationWrapper, depth=False)]
    envs = make_eval_envs(args.env_id, args.num_eval_envs, args.sim_backend, env_kwargs, other_kwargs, video_dir=f'runs/{run_name}/videos' if args.capture_video else None, wrappers=wrappers)

    if args.track:
        import wandb
        config = vars(args)
        config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, env_horizon=gym_utils.find_max_episode_steps_value(envs))
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            group="ACT",
            tags=["act"]
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # dataloader setup
    dataset = SmallDemoDataset_ACTPolicy(args.demo_path, args.num_queries, num_traj=args.num_demos)
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

    # optimizer setup
    param_dicts = [
        {"params": [p for n, p in agent.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in agent.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = optim.AdamW(param_dicts, lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6)

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

    # Evaluation
    eval_kwargs = dict(
        stats=dataset.norm_stats, num_queries=args.num_queries, temporal_agg=args.temporal_agg,
        max_timesteps=gym_utils.find_max_episode_steps_value(envs), device=device, sim_backend=args.sim_backend
    )

    # ---------------------------------------------------------------------------- #
    # Training begins.
    # ---------------------------------------------------------------------------- #
    agent.train()

    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)

    for iteration, data_batch in enumerate(train_dataloader):
        # copy data from cpu to gpu
        obs_batch_dict = data_batch['observations']
        obs_batch_dict = {k: v.cuda(non_blocking=True) for k, v in obs_batch_dict.items()}
        act_batch = data_batch['actions'].cuda(non_blocking=True)

        # forward and compute loss
        loss_dict = agent.compute_loss(
            obs=obs_batch_dict, # obs_batch_dict['state'] is (B, obs_dim)
            action_seq=act_batch, # (B, num_queries, act_dim)
        )
        total_loss = loss_dict['loss']  # total_loss = l1 + kl * self.kl_weight

        # backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step() # step lr scheduler every batch, this is different from standard pytorch behavior
        last_tick = time.time()

        # update Exponential Moving Average of the model weights
        ema.step(agent.parameters())
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if iteration % args.log_freq == 0:
            print(f"Iteration {iteration}, loss: {total_loss.item()}")
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], iteration)
            writer.add_scalar("charts/backbone_learning_rate", optimizer.param_groups[1]["lr"], iteration)
            writer.add_scalar("losses/total_loss", total_loss.item(), iteration)
            for k, v in timings.items():
                writer.add_scalar(f"time/{k}", v, iteration)

        # Evaluation
        if iteration % args.eval_freq == 0:
            last_tick = time.time()

            ema.copy_to(ema_agent.parameters())

            eval_metrics = evaluate(args.num_eval_episodes, ema_agent, envs, eval_kwargs)
            timings["eval"] += time.time() - last_tick

            print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], iteration)
                print(f"{k}: {eval_metrics[k]:.4f}")

            save_on_best_metrics = ["success_once", "success_at_end"]
            for k in save_on_best_metrics:
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = eval_metrics[k]
                    save_ckpt(run_name, f"best_eval_{k}")
                    print(f'New best {k}_rate: {eval_metrics[k]:.4f}. Saving checkpoint.')
        
        # Checkpoint
        if args.save_freq is not None and iteration % args.save_freq == 0:
            save_ckpt(run_name, str(iteration))

    envs.close()
    writer.close()
