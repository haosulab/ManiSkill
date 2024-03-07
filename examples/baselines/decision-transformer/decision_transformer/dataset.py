import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from mani_skill.utils.io_utils import load_json


def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


class ManiSkill2Dataset(Dataset):
    def __init__(self, dataset_file: str, load_count=-1) -> None:
        self.dataset_file = dataset_file
        # for details on how the code below works, see the
        # quick start tutorial
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]

        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]

        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.total_frames = 0
        if load_count == -1:
            load_count = len(self.episodes)
        for eps_id in tqdm(range(load_count)):
            eps = self.episodes[eps_id]
            trajectory = self.data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)

            # we use :-1 here to ignore the last observation as that
            # is the terminal observation which has no actions
            self.observations.append(trajectory["obs"][:-1])
            self.actions.append(trajectory["actions"])
            self.rewards.append(trajectory["rewards"][1:])
            self.dones.append(trajectory["success"])
            # print(trajectory.keys())

        # self.rewards = np.vstack(self.rewards)

    def get_state_stats(self):
        arr = np.vstack(self.observations)
        return np.mean(arr, axis=0), np.std(arr, axis=0) + 1e-6

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        action = torch.from_numpy(self.actions[idx]).float()
        obs = torch.from_numpy(self.observations[idx]).float()
        rew = torch.from_numpy(self.rewards[idx]).float()
        done = torch.from_numpy(self.dones[idx]).float()
        return obs, action, rew, done


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def load_batch(
    dataset: ManiSkill2Dataset,
    batch_size=16,
    max_len=20,
    num_trajectories=100,
    state_dim=42,
    act_dim=7,
    max_ep_len=90,
    scale=50,
    device="cuda",
):
    batch_inds = np.random.choice(
        np.arange(len(dataset)),
        size=batch_size,
        replace=True,
    )
    state_mean, state_std = dataset.get_state_stats()
    s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
    for i in range(batch_size):
        obs, action, rew, done = dataset[batch_inds[i]]
        si = random.randint(0, rew.shape[0] - 1 - max_len)
        s.append(obs[si : si + max_len].reshape(1, -1, state_dim))
        a.append(action[si : si + max_len].reshape(1, -1, act_dim))
        r.append(rew[si : si + max_len].reshape(1, -1, 1))
        d.append(done[si : si + max_len].reshape(1, -1))
        timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
        timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1
        rtg.append(
            discount_cumsum(rew[si:], gamma=1.0)[: s[-1].shape[1] + 1].reshape(1, -1, 1)
        )
        tlen = s[-1].shape[1]
        s[-1] = np.concatenate(
            [np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1
        )
        s[-1] = (s[-1] - state_mean) / state_std
        a[-1] = np.concatenate(
            [np.ones((1, max_len - tlen, act_dim)) * -10.0, a[-1]], axis=1
        )
        r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
        d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
        rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1)
        timesteps[-1] = np.concatenate(
            [np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1
        )
        mask.append(
            np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1)
        )

    s = torch.from_numpy(np.concatenate(s, axis=0)).to(
        dtype=torch.float32, device=device
    )
    a = torch.from_numpy(np.concatenate(a, axis=0)).to(
        dtype=torch.float32, device=device
    )
    r = torch.from_numpy(np.concatenate(r, axis=0)).to(
        dtype=torch.float32, device=device
    )
    d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
    rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(
        dtype=torch.float32, device=device
    )
    timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
        dtype=torch.long, device=device
    )
    mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
    return s, a, r, d, rtg, timesteps, mask
