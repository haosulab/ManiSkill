import argparse

import gymnasium as gym
import torch
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusion_policy.dataset import ManiSkill2Dataset
from diffusion_policy.model import ConditionalUnet1D
from diffusion_policy.training import train

import mani_skill.envs

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    type=str,
    help="Specifies the env to run the training",
)
parser.add_argument("--dataset", type=str)
parser.add_argument("--pred_horizon", type=int, default=16)
parser.add_argument("--obs_horizon", type=int, default=2)
parser.add_argument("--action_horizon", type=int, default=8)
parser.add_argument("--num_eval_eps", type=int, default=10)
parser.add_argument("--eval_ep_len", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--wandb", type=bool, default=False)
parser.add_argument("--wandb_key", type=str)
parser.add_argument("--n_epochs", type=int, default=100000)
parser.add_argument("--eval_interval", type=int, default=1000)
parser.add_argument("--num_diffusion_iters", type=int, default=100)
parser.add_argument("--save_weights", type=bool, default=False)
parser.add_argument("--video", type=bool, default=False)


args = parser.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
config = vars(args)
env_id = config["env"]
config["device"] = device
env = gym.make(
    env_id,
    obs_mode="state",
    control_mode="pd_ee_delta_pose",
    render_mode="cameras",
    enable_shadow=True,
)

dataset = ManiSkill2Dataset(config=config)
config["obs_dim"] = dataset.obs_space
config["action_dim"] = dataset.action_space

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    num_workers=1,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process after each epoch
    persistent_workers=True,
    generator=torch.Generator(device),
)

noise_pred_net = ConditionalUnet1D(
    input_dim=config["action_dim"],
    global_cond_dim=config["obs_dim"] * config["obs_horizon"],
)
noise_scheduler = DDPMScheduler(
    num_train_timesteps=config["num_diffusion_iters"],
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule="squaredcos_cap_v2",
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type="epsilon",
)

ema = EMAModel(parameters=noise_pred_net.parameters(), power=0.75)

optimizer = torch.optim.AdamW(
    params=noise_pred_net.parameters(), lr=1e-4, weight_decay=1e-6
)


lr_scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * config["n_epochs"],
)


if config["wandb"]:
    import random

    import wandb

    wandb.login(key=config["wandb_key"])

    wandb.init(
        name=f"maniskill2-{env_id}-{random.randint(0, 100000)}",
        group=f"maniskill2-{env_id}",
        project="diffusion-policy",
        config=config,
    )


train(
    noise_pred_net,
    optimizer,
    noise_scheduler,
    dataloader,
    ema,
    lr_scheduler,
    env,
    config,
    device,
)
