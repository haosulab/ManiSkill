import gymnasium as gym
import torch
import numpy as np

class TensorWrapper(gym.Wrapper):
    """
    Wrapper for converting numpy arrays to torch tensors.
    """

    def __init__(self, env):
        super().__init__(env)

    def rand_act(self):
        return torch.from_numpy(self.action_space.sample().astype(np.float32))

    def _try_f32_tensor(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
            if x.dtype == torch.float64:
                x = x.float()
        return x

    def _obs_to_tensor(self, obs):
        if isinstance(obs, dict):
            for k in obs.keys():
                obs[k] = self._try_f32_tensor(obs[k])
        else:
            obs = self._try_f32_tensor(obs)
        return obs

    def _info_to_tensor(self, info):
        tensor_dict = {}
        for key, value in info.items():
            if isinstance(value, np.ndarray):
                try:
                    tensor_dict[key] = torch.from_numpy(np.stack(value))
                except Exception as e:
                    tensor_dict[key] = value
            else:
                tensor_dict[key] = value
        return tensor_dict

    def reset(self, task_idx=None):
        obs, info = self.env.reset()
        return self._obs_to_tensor(obs), self._info_to_tensor(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action.numpy())
        return self._obs_to_tensor(obs), torch.tensor(reward, dtype=torch.float32), torch.tensor(terminated), torch.tensor(truncated), self._info_to_tensor(info)
