import gymnasium as gym
import numpy as np
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter
import mani_skill

class DictArray(object):
    """Utility wrapper that stores a dict of torch tensors with the same leading buffer shape."""

    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    dtype = (
                        torch.float32 if v.dtype in (np.float32, np.float64) else
                        torch.uint8 if v.dtype == np.uint8 else
                        torch.int16 if v.dtype == np.int16 else
                        torch.int32 if v.dtype == np.int32 else
                        torch.bool if v.dtype in (np.bool_, bool) else
                        v.dtype
                    )
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {k: v[index] for k, v in self.data.items()}

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k, v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)


def build_checkpoint(agent, args, envs):
    """
    Pack everything you might need at inference time into one dict.
    """
    ckpt = {
        "model": agent.state_dict(),          
        "obs_rms": getattr(envs, "obs_rms", None),
        "cfg": vars(args),                    
        "meta": {
            "torch": torch.__version__,
            "mani_skill": mani_skill.__version__,
        },
    }
    return ckpt

class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)
    def close(self):
        self.writer.close()

# -----------------------------------------------------------------------------
# GridSampler: ensure shuffled, non-repeating sampling across training batches
# -----------------------------------------------------------------------------
class GridSampler:
    def __init__(self, all_grids, batch_size: int):
        self.all_grids = all_grids
        self.total_grids = len(all_grids)
        self.batch_size = batch_size
        self.indices = np.array([], dtype=int)

    def _ensure_indices(self):
        if len(self.indices) < self.batch_size:
            new_indices = np.arange(self.total_grids)
            np.random.shuffle(new_indices)
            self.indices = np.concatenate([self.indices, new_indices])

    def sample(self):
        self._ensure_indices()
        batch_indices = self.indices[: self.batch_size]
        self.indices = self.indices[self.batch_size :]
        sampled_grids = [self.all_grids[i] for i in batch_indices]
        return batch_indices, sampled_grids