from dataclasses import dataclass
import torch
import os
import gymnasium as gym
import numpy as np

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.trajectory.dataset import load_h5_data
from mani_skill.utils import common, gym_utils
import h5py

@dataclass
class ReverseCurriculumConfig:
    reverse_curriculum_sampler: str = "uniform"
    demo_horizon_to_max_steps_ratio: float = 3.0
    max_steps_min: int = 8
    per_demo_buffer_size: int = 3
    reverse_step_size: int = 1
    traj_ids: list[str] = None

class ReverseCurriculumWrapper(gym.Wrapper):
    """Apply this before any auto reset wrapper"""
    def __init__(self, env, dataset_path,
                 cfg = ReverseCurriculumConfig(),
                 eval_mode=False):
        super().__init__(env)
        self.curriculum_mode = "reverse"
        """which curriculum to apply to modify env states during training"""
        self.eval_mode = eval_mode
        self.cfg = cfg

        # Reverse curriculum specific configs
        self.reverse_curriculum_sampler = self.cfg.reverse_curriculum_sampler
        self.demo_horizon_to_max_steps_ratio = self.cfg.demo_horizon_to_max_steps_ratio
        self.max_steps_min = self.cfg.max_steps_min
        self.per_demo_buffer_size = self.cfg.per_demo_buffer_size
        self.reverse_step_size = self.cfg.reverse_step_size

        # load dataset
        dataset_path = os.path.expanduser(dataset_path)
        h5py_file = h5py.File(dataset_path, "r")
        max_eps_len = -1
        env_states_flat_list = []
        traj_ids = self.cfg.traj_ids
        if traj_ids is None:
            traj_ids = list(h5py_file.keys())
        self.traj_count = len(traj_ids)
        for traj_id in traj_ids:
            env_states = load_h5_data(h5py_file[traj_id]["env_states"])
            env_states_flat = common.flatten_state_dict(env_states)
            max_eps_len = max(len(env_states_flat), max_eps_len)
            env_states_flat_list.append(env_states_flat)

        self.env_states = torch.zeros((self.traj_count, max_eps_len, env_states_flat.shape[-1]), device=self.base_env.device)
        """environment states flattened into a matrix of shape (B, T, D) where B is the number of episodes, T is the maximum episode length, and D is the dimension of the state"""
        self.demo_curriculum_step = torch.zeros((self.traj_count,), dtype=torch.int32)
        """the current curriculum step/stage for each demonstration given. Used in reverse curricula options"""
        self.demo_horizon = torch.zeros((self.traj_count,), dtype=torch.int32, device=self.base_env.device)
        """length of each demo"""
        self.demo_solved = torch.zeros((self.traj_count,), dtype=torch.bool, device=self.base_env.device)
        """whether the demo is solved (meaning the curriculum stage has reached the end)"""
        for i, env_states_flat in enumerate(env_states_flat_list):
            self.env_states[i, :len(env_states_flat)] = torch.from_numpy(env_states_flat).float().to(self.base_env.device)
            self.demo_horizon[i] = len(env_states_flat)
        if not self.eval_mode:
            self.demo_curriculum_step = self.demo_horizon - 1
        h5py_file.close()

        self._demo_success_rate_buffer_pos = torch.zeros((self.traj_count, ), dtype=torch.int, device=self.base_env.device)
        self.demo_success_rate_buffers = torch.zeros((self.traj_count, self.per_demo_buffer_size), dtype=torch.bool, device=self.base_env.device)
        self.max_episode_steps = gym_utils.find_max_episode_steps_value(self.env)
        self.sampled_traj_indexes = torch.zeros((self.base_env.num_envs, ), dtype=torch.int, device=self.base_env.device)
        self.dynamic_max_episode_steps = torch.zeros((self.base_env.num_envs, ), dtype=torch.int, device=self.base_env.device)
        self._traj_index_density = np.ones((self.traj_count, ), dtype=np.float32) / self.traj_count
        assert self.max_episode_steps is not None, "Reverse curriculum wrapper requires max_episode_steps to be set"
        print(f"ReverseForwardCurriculumWrapper initialized. Loaded {self.traj_count} demonstrations. Trajectory IDs: {traj_ids} \n \
              Mean Length: {np.mean(self.demo_horizon.cpu().numpy())}, \
              Max Length: {np.max(self.demo_horizon.cpu().numpy())}")
    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.curriculum_mode == "reverse":
            truncated: torch.Tensor = (
                self.base_env.elapsed_steps >= self.dynamic_max_episode_steps
            )
        else:
            truncated: torch.Tensor = (
                self.base_env.elapsed_steps >= self.max_episode_steps
            )
            assert truncated.any() == truncated.all()
        assert "success" in info, "Reverse curriculum wrapper currently requires there to be a success key in the info dict"
        if not self.eval_mode:
            if self.curriculum_mode == "reverse" and truncated.any():
                truncated_traj_idxs = self.sampled_traj_indexes[truncated]
                self.demo_success_rate_buffers[truncated_traj_idxs, self._demo_success_rate_buffer_pos[truncated_traj_idxs]] = info["success"][truncated]

                # advance curriculum. code below is indexing arrays shaped by the number of demos
                self._demo_success_rate_buffer_pos[truncated_traj_idxs] = (self._demo_success_rate_buffer_pos[truncated_traj_idxs] + 1) % self.per_demo_buffer_size
                per_demo_success_rates = self.demo_success_rate_buffers.float().mean(dim=1)
                can_advance = per_demo_success_rates > 0.9
                self.demo_curriculum_step[can_advance] -= self.reverse_step_size
                self.demo_success_rate_buffers[can_advance, :] = 0
                self.demo_solved[self.demo_curriculum_step < 0] = True
                self.demo_curriculum_step = torch.clamp(self.demo_curriculum_step, 0)

                if can_advance.any():
                    # update the probability distribution for trajectory sampling
                    self._traj_index_density = torch.divide(self.demo_curriculum_step, self.demo_horizon)
                    self._traj_index_density[self.demo_solved] = 1e-6
                    self._traj_index_density = (self._traj_index_density / self._traj_index_density.sum()).cpu().numpy()

        return obs, reward, terminated, truncated, info
    def reset(self, *, seed=None, options=dict()):
        super().reset(seed=seed, options=options)
        if "env_idx" in options:
            env_idx = options["env_idx"]
        else:
            env_idx = torch.arange(0, self.base_env.num_envs, device=self.base_env.device)
        if self.curriculum_mode == "reverse":
            # set initial state accordingly
            b = len(env_idx)
            self.sampled_traj_indexes[env_idx] = torch.from_numpy(
                self.base_env._episode_rng.choice(np.arange(0, self.traj_count), size=(b, ), replace=True, p=self._traj_index_density)
            ).int().to(self.base_env.device)
            reset_traj_indexes = self.sampled_traj_indexes[env_idx]
            if self.eval_mode:
                self.base_env.set_state(self.env_states[self.sampled_traj_indexes, 5 + torch.zeros((b, ), dtype=torch.int, device=self.base_env.device)], env_idx)
            elif self.reverse_curriculum_sampler == "geometric":
                x_start_steps_density_list = [0.5, 0.25, 0.125, 0.125 / 2, 0.125 / 2]
                sampled_offsets = torch.from_numpy(self.base_env._episode_rng.randint(0, len(x_start_steps_density_list), size=(b, ))).to(self.base_env.device)
                x_start_steps = self.demo_curriculum_step[reset_traj_indexes] + sampled_offsets
                x_start_steps = torch.clamp(x_start_steps, torch.zeros((b, ), device=self.base_env.device), self.demo_horizon[reset_traj_indexes] - 1).int()
            self.dynamic_max_episode_steps[env_idx] = 8 + (self.demo_horizon[reset_traj_indexes] - x_start_steps) // self.demo_horizon_to_max_steps_ratio
            self.base_env.set_state(self.env_states[reset_traj_indexes, x_start_steps], env_idx)
        obs = self.base_env.get_obs()
        return obs, {}


@dataclass
class ForwardCurriculumConfig:
    forward_curriculum: str = "success_once_score"
    staleness_coef: float =  0.1
    staleness_temperature: float =  0.1
    staleness_transform: str = "rankmin"
    score_transform: str = "rankmin"
    score_temperature: float = 0.1
