import collections

import numpy as np
import torch

from .model import ConditionalUnet1D


def evaluate(model: ConditionalUnet1D, env, noise_scheduler, config, device):
    r = []
    s = []
    model.eval()

    for _ in range(config["num_eval_eps"]):
        obs, info = env.reset()
        rewards = list()
        obs_deque = collections.deque(
            [obs] * config["obs_horizon"], maxlen=config["obs_horizon"]
        )

        steps = 0
        while steps < config["eval_ep_len"]:
            obs_seq = np.stack(obs_deque)

            with torch.no_grad():
                noisy_action = torch.randn(
                    (1, config["pred_horizon"], config["action_dim"]),
                    device=config["device"],
                )
                obs = torch.from_numpy(obs_seq).to(device=device, dtype=torch.float32)
                obs = obs.unsqueeze(0).flatten(1)
                noise_scheduler.set_timesteps(config["num_diffusion_iters"])
                for k in noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = model(sample=noisy_action, timestep=k, global_cond=obs)

                    # inverse diffusion step (remove noise)
                    noisy_action = noise_scheduler.step(
                        model_output=noise_pred, timestep=k, sample=noisy_action
                    ).prev_sample
                actions = noisy_action[0].detach().to(device="cpu").numpy()
                start = config["obs_horizon"] - 1
                end = start + config["action_horizon"]

                for action in actions[start:end]:
                    observation, reward, _, _, info = env.step(action)
                    obs_deque.append(observation)
                    # and reward/vis
                    rewards.append(reward)
                    steps += 1
        s.append(info["success"])
        r.append(sum(rewards) / len(rewards))
    model.train()
    return s, r
