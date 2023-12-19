import numpy as np
import torch


def evaluate_episode_rtg(
    env,
    state_dim,
    act_dim,
    model,
    max_ep_len=100,
    scale=10.0,
    state_mean=0.0,
    state_std=1.0,
    device="cuda:0",
    target_return=None,
    mode="normal",
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()[0]
    if mode == "noise":
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = (
        torch.from_numpy(state)
        .reshape(1, state_dim)
        .to(device=device, dtype=torch.float32)
    )
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(
        1, 1
    )
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != "delayed":
            pred_return = target_return[0, -1] - (reward / scale)
        else:
            pred_return = target_return[0, -1]
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)],
            dim=1,
        )

        episode_return += reward
        episode_length += 1

        # if done:
        #     break
    success = info["success"]
    return episode_return, episode_length, success


def eval_episodes(target_rew, env, variant):
    def fn(model):
        returns, lengths = [], []
        success = 0
        for _ in range(variant["num_eval_episodes"]):
            with torch.no_grad():
                ret, length, s = evaluate_episode_rtg(
                    env,
                    variant["state_dim"],
                    variant["act_dim"],
                    model,
                    max_ep_len=variant["max_ep_len"],
                    scale=10,
                    target_return=target_rew / 10,
                    mode="normal",
                    state_mean=variant["state_mean"],
                    state_std=variant["state_std"],
                    device="cuda:0",
                )
            returns.append(ret)
            lengths.append(length)
            success += s
        return {
            f"target_{target_rew}_return_mean": np.mean(returns),
            f"target_{target_rew}_return_std": np.std(returns),
            f"target_{target_rew}_length_mean": np.mean(lengths),
            f"target_{target_rew}_length_std": np.std(lengths),
            f"target_{target_rew}_success_rate": success / len(returns),
        }

    return fn
