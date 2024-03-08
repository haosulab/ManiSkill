import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from mani_skill.trajectory.utils import dict_to_list_of_dicts
from mani_skill.utils.io_utils import load_json


# loads h5 data into memory for faster access
def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


class ManiSkillTrajectoryDataset(Dataset):
    """
    A general torch Dataset you can drop in and use immediately with just about any trajectory .h5 data generated from ManiSkill.
    This class simply is a simple starter code to load trajectory data easily, but does not do any data transformation or anything
    advanced. We recommend you to copy this code directly and modify it for more advanced use cases

    Args:
        dataset_file (str): path to the .h5 file containing the data you want to load
        load_count (int): the number of trajectories from the dataset to load into memory. If -1, will load all into memory
    """

    def __init__(self, dataset_file: str, load_count=-1) -> None:
        self.dataset_file = dataset_file

        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]

        self.obs_state = []
        self.obs_rgbd = []
        self.actions = []
        self.total_frames = 0
        if load_count == -1:
            load_count = len(self.episodes)
        for eps_id in tqdm(range(load_count)):
            eps = self.episodes[eps_id]
            trajectory = self.data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)
            self.obs = trajectory["obs"]
            # we use :-1 to ignore the last obs as terminal observations are included
            # and they don't have actions
            # self.obs_rgbd.append(obs["rgbd"][:-1])
            # self.obs_state.append(obs["state"][:-1])
            self.actions.append(trajectory["actions"])
        self.obs_rgbd = np.vstack(self.obs_rgbd)
        self.obs_state = np.vstack(self.obs_state)
        self.actions = np.vstack(self.actions)

    def __len__(self):
        return len(self.obs_rgbd)

    def __getitem__(self, idx):
        action = torch.from_numpy(self.actions[idx]).float()
        rgbd = self.obs_rgbd[idx]
        # note that we rescale data on demand as opposed to storing the rescaled data directly
        # so we can save a ton of space at the cost of a little extra compute
        rgbd = rescale_rgbd(rgbd)
        # permute data so that channels are the first dimension as PyTorch expects this
        rgbd = torch.from_numpy(rgbd).float().permute((2, 0, 1))
        state = torch.from_numpy(self.obs_state[idx]).float()
        return dict(rgbd=rgbd, state=state), action
