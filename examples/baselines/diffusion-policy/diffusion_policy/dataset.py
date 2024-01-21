import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from mani_skill2.utils.io_utils import load_json


def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


def create_sample_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        # if i > 0:
        #     start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx, i]
            )
    indices = np.array(indices)
    return indices


def sample_sequence(
    data, seq_len, buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx, i
):
    sample = data[i][buffer_start_idx:buffer_end_idx]
    if sample_start_idx > 0:
        sample = np.insert(sample, 0, np.tile(sample[0], (sample_start_idx, 1)), axis=0)
    if sample_end_idx < seq_len:
        sample = np.insert(
            sample, -1, np.tile(sample[-1], (seq_len - sample_end_idx, 1)), axis=0
        )
    return sample


class ManiSkill2Dataset(Dataset):
    def __init__(self, config, load_count=-1) -> None:
        self.dataset_file = config["dataset"]
        # for details on how the code below works, see the
        # quick start tutorial
        self.data = h5py.File(config["dataset"], "r")
        json_path = config["dataset"].replace(".h5", ".json")
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
            # print(trajectory.keys())
        ends = [action.shape[0] for action in self.actions]
        self.action_space = self.actions[0].shape[-1]
        self.obs_space = self.observations[0].shape[-1]

        self.episode_ends = np.array(ends)
        self.inds = create_sample_indices(
            self.episode_ends,
            config["pred_horizon"],
            config["obs_horizon"] - 1,
            config["action_horizon"] - 1,
        )
        self.pred_horizon = config["pred_horizon"]
        self.obs_horizon = config["obs_horizon"]

        # self.rewards = np.vstack(self.rewards)

    def get_state_stats(self):
        arr = np.vstack(self.observations)
        return np.mean(arr, axis=0), np.std(arr, axis=0) + 1e-6

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):

        action = sample_sequence(self.actions, self.pred_horizon, *self.inds[idx])
        obs = sample_sequence(self.observations, self.pred_horizon, *self.inds[idx])

        return torch.from_numpy(action).to(torch.float32), torch.from_numpy(
            obs[: self.obs_horizon, :]
        ).to(torch.float32)
