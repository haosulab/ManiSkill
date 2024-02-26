import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from mani_skill2.utils.wrappers import RecordEpisode

from .evaluate import evaluate
from .model import ConditionalUnet1D


def train(
    noise_pred_net: ConditionalUnet1D,
    optimizer,
    noise_scheduler,
    dataloader,
    ema,
    lr_scheduler,
    env,
    config: dict,
    device: str,
):
    if config["wandb"]:
        import wandb

        def log_fn(output: dict):
            wandb.log(output)

    log = {}
    noise_pred_net.train()
    for epoch in tqdm(range(config["n_epochs"])):
        epoch_loss = []
        for batch in dataloader:
            obs = batch[1].to(device=device)
            action = batch[0].to(device=device)
            B = obs.shape[0]
            obs_cond = obs[:, : config["obs_horizon"], :]
            # (B, obs_horizon * obs_dim)
            obs_cond = obs_cond.flatten(start_dim=1)
            noise = torch.randn(action.shape, device=device)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (B,), device=device
            ).long()

            # add noise to the clean images according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = noise_scheduler.add_noise(action, noise, timesteps)

            # predict the noise residual
            noise_pred = noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)

            # L2 loss
            loss = nn.functional.mse_loss(noise_pred, noise)

            # optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # step lr scheduler every batch
            # this is different from standard pytorch behavior
            lr_scheduler.step()
            ema.step(noise_pred_net.parameters())

            # logging
            loss_cpu = loss.item()
            epoch_loss.append(loss_cpu)

        if epoch % config["eval_interval"] == 0 and epoch > 0:
            ema_noise_pred_net = noise_pred_net
            ema.copy_to(ema_noise_pred_net.parameters())
            log["training/epoch_loss_mean"] = np.mean(np.array(epoch_loss))
            log["training/epoch_loss_std"] = np.std(np.array(epoch_loss))
            s, r = evaluate(
                ema_noise_pred_net, env, noise_scheduler, config, config["device"]
            )
            log["eval/success_rate"] = sum(np.array(s)) / len(s)
            log["eval/reward_avg"] = np.mean(np.array(r))
            log["eval/reward_std"] = np.std(np.array(r))
            if config["wandb"]:
                log_fn(log)
    ema_noise_pred_net = noise_pred_net
    ema.copy_to(ema_noise_pred_net.parameters())

    if config["video"]:
        video_env = RecordEpisode(env, "./videos", info_on_video=True)
        evaluate(
            ema_noise_pred_net, video_env, noise_scheduler, config, config["device"]
        )
        video_env.flush_video()
        if config["wandb"]:
            wandb.log({f"video_{config['env']}": wandb.Video("./videos/0.mp4")})

    if config["save_weights"]:
        torch.save(
            ema_noise_pred_net.state_dict(),
            os.path.join(wandb.run.dir, f"weights_{config['env']}.pt"),
        )
        if config["wandb"]:
            wandb.save(f"weights_{config['env']}.pt")
