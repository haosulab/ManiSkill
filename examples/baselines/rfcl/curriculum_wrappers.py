from dataclasses import dataclass, field
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
from collections import deque, defaultdict
@dataclass
class DemoCurriculumMetadata:
    start_step: int = None  # t_i
    total_steps: int = None  # T_i
    success_rate_buffer = deque(maxlen=2)  # size of this is m
    episode_steps_back = deque(maxlen=2)
    solved: bool = False  # whether we have reverse solved this demo

def create_filled_deque(maxlen, fill_value):
    return deque([fill_value] * maxlen, maxlen=maxlen)
class ReverseCurriculumWrapper(gym.Wrapper):
    """Apply this before any auto reset wrapper"""
    def __init__(self, env, dataset_path,
                 cfg = ReverseCurriculumConfig(),
                 ignore_terminations: bool = True,
                 eval_mode=False):
        super().__init__(env)
        self.curriculum_mode = "reverse"
        """which curriculum to apply to modify env states during training"""
        self.eval_mode = eval_mode
        self.cfg = cfg
        self.ignore_terminations = ignore_terminations
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
            if isinstance(traj_id, int): traj_id = f"traj_{traj_id}"
            env_states = load_h5_data(h5py_file[traj_id]["env_states"])
            env_states_flat = common.flatten_state_dict(env_states)
            max_eps_len = max(len(env_states_flat), max_eps_len)
            env_states_flat_list.append(env_states_flat)
        self.env_states = torch.zeros((self.traj_count, max_eps_len, env_states_flat.shape[-1]), device=self.base_env.device)
        """environment states flattened into a matrix of shape (B, T, D) where B is the number of episodes, T is the maximum episode length, and D is the dimension of the state"""
        self.demo_metadata = defaultdict(DemoCurriculumMetadata)


        # self.demo_curriculum_step = torch.zeros((self.traj_count,), dtype=torch.int32)
        # """the current curriculum step/stage for each demonstration given. Used in reverse curricula options"""
        self.demo_horizon = torch.zeros((self.traj_count,), dtype=torch.int32, device=self.base_env.device)
        """length of each demo"""
        # self.demo_solved = torch.zeros((self.traj_count,), dtype=torch.bool, device=self.base_env.device)
        # """whether the demo is solved (meaning the curriculum stage has reached the end)"""
        # @dataclass
        # class DemoMeta:
        #     success: list[int] = field(default_factory=list)
        #     count: list[int] = field(default_factory=list)
        # self.demo_success_rate_buffers: dict[str, DemoMeta] = dict()

        for i, env_states_flat in enumerate(env_states_flat_list):
            self.env_states[i, :len(env_states_flat)] = torch.from_numpy(env_states_flat).float().to(self.base_env.device)
            self.demo_horizon[i] = len(env_states_flat)
            # self.demo_success_rate_buffers[i] = DemoMeta()

            start_step = len(env_states_flat) - 1
            self.demo_metadata[i].start_step = start_step
            self.demo_metadata[i].total_steps =  len(env_states_flat)
            self.demo_metadata[i].success_rate_buffer = create_filled_deque(self.per_demo_buffer_size, 0)
            self.demo_metadata[i].episode_steps_back = create_filled_deque(self.per_demo_buffer_size, -1)
        # if not self.eval_mode:
        #     self.demo_curriculum_step = self.demo_horizon - 1
        h5py_file.close()

        # self._demo_success_rate_buffer_pos = torch.zeros((self.traj_count, ), dtype=torch.int, device=self.base_env.device)
        # self.demo_success_rate_buffers = torch.zeros((self.traj_count, self.per_demo_buffer_size), dtype=torch.bool, device=self.base_env.device)



        self.max_episode_steps = gym_utils.find_max_episode_steps_value(self.env)
        self.sampled_traj_indexes = torch.zeros((self.base_env.num_envs, ), dtype=torch.int, device=self.base_env.device)
        self.sampled_start_steps = torch.zeros((self.base_env.num_envs, ), dtype=torch.int, device=self.base_env.device)
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
            ) | (
                self.base_env.elapsed_steps >= self.max_episode_steps
            )
        else:
            truncated: torch.Tensor = (
                self.base_env.elapsed_steps >= self.max_episode_steps
            )
        if self.ignore_terminations:
            dones = truncated
        else:
            dones = terminated | truncated
        assert "success" in info, "Reverse curriculum wrapper currently requires there to be a success key in the info dict"
        if not self.eval_mode:
            if self.curriculum_mode == "reverse" and dones.any():

                for i in range(len(dones)):
                    if dones[i].item():
                        traj_idx = self.sampled_traj_indexes[i].item()
                        metadata = self.demo_metadata[traj_idx]
                        if self.sampled_start_steps[i].item() == metadata.start_step:
                            metadata.success_rate_buffer.append(int(info["success"][i].item()))
                            # metadata.episode_steps_back.append(final_info["steps_back"])
                            # self.global_success_rate_history.append(int(success))
                # curr advancing
                change=False
                for demo_id in range(self.traj_count):
                    metadata = self.demo_metadata[demo_id]
                    running_success_rate_mean = np.mean(metadata.success_rate_buffer)
                    if running_success_rate_mean >= 1.0:
                        metadata.success_rate_buffer = create_filled_deque(self.per_demo_buffer_size, 0)
                        metadata.episode_steps_back = create_filled_deque(self.per_demo_buffer_size, -1)
                        if metadata.start_step > 0:
                            metadata.start_step = max(metadata.start_step - self.reverse_step_size, 0)
                            if True:
                                print(f"Demo {demo_id} stepping back to {metadata.start_step}")
                            change = True
                        else:
                            if not metadata.solved:
                                if True:
                                    print(f"Demo {demo_id} is reverse solved!")
                                metadata.solved = True
                                change = True
                if change:
                    for demo_id in range(self.traj_count):
                        self._traj_index_density[demo_id] = self.demo_metadata[demo_id].start_step / self.demo_metadata[demo_id].total_steps
                        if self.demo_metadata[demo_id].start_step == 0:
                            self._traj_index_density[demo_id] = 1e-6
                    # self._traj_index_density[self.demo_curriculum_step == 0] = 1e-6
                    self._traj_index_density = (self._traj_index_density / self._traj_index_density.sum())

                # mask = dones & (self.demo_curriculum_step[self.sampled_traj_indexes] >= self.sampled_start_steps)
                # traj_idxs_to_check = self.sampled_traj_indexes[mask]
                # can_advance = torch.zeros((self.traj_count, ), dtype=torch.bool, device=self.base_env.device)
                # successes_of_traj_to_check = info["success"][mask]
                # # impl below is the same as the original and is a for loop
                # traj_idxs_to_check = traj_idxs_to_check.cpu().numpy()
                # for success, traj_idx in zip(successes_of_traj_to_check, traj_idxs_to_check):
                #     self.demo_success_rate_buffers[traj_idx].success.append(success.item())
                #     if len(self.demo_success_rate_buffers[traj_idx].success) > self.per_demo_buffer_size:
                #         self.demo_success_rate_buffers[traj_idx].success.pop(0)
                # for traj_idx in range(self.traj_count):
                #     metadata = self.demo_success_rate_buffers[traj_idx]
                #     if len(metadata.success) > 0 and np.mean(metadata.success) >= 1.0:
                #         can_advance[traj_idx] = True
                #         metadata.success = []

                # # alternative impl is more parallelized
                # # for traj_idx in range(self.traj_count):
                # #     traj_mask = traj_idxs_to_check == traj_idx
                # #     matched = traj_mask.sum()
                # #     if matched > 0:
                # #         # successes_of_traj_idx = successes_of_traj_to_check[traj_mask]
                # #         # if successes_of_traj_idx.any():
                # #         #     for i, success in enumerate(successes_of_traj_idx):
                # #         #         self.demo_success_rate_buffers[traj_idx].success.append(success.item())
                # #         #         if len(self.demo_success_rate_buffers[traj_idx].success) > 3:
                # #         #             self.demo_success_rate_buffers[traj_idx].success.pop(0)
                # #         #     if np.mean(self.demo_success_rate_buffers[traj_idx].success) >= 1.0:
                # #         #         can_advance[traj_idx] = True
                # #         #         # self.demo_success_rate_buffers[traj_idx].count.append(1)
                # #         #     # import ipdb;ipdb.set_trace()
                # #         metadata = self.demo_success_rate_buffers[traj_idx]
                # #         metadata.success.append(successes_of_traj_to_check[traj_mask].sum().item())
                # #         metadata.count.append(matched.item())
                # #         tc = np.sum(metadata.count)
                # #         if tc > self.per_demo_buffer_size:
                # #             sr_val = np.sum(metadata.success) / tc
                # #             if sr_val >= 0.9:
                # #                 can_advance[traj_idx] = True
                # #                 metadata.success = []
                # #                 metadata.count = []
                # #             else:
                # #                 trim_idx = 0
                # #                 ct = 0
                # #                 for i in range(len(metadata.count)):
                # #                     ct += metadata.count[i]
                # #                     if ct >= self.per_demo_buffer_size:
                # #                         trim_idx = i
                # #                 metadata.success = metadata.success[trim_idx:]
                # #                 metadata.count = metadata.count[trim_idx:]
                # # advance curriculum. code below is indexing arrays shaped by the number of demos
                # self.demo_curriculum_step[can_advance] -= self.reverse_step_size
                # # self.demo_success_rate_buffers[can_advance, :] = 0
                # self.demo_solved[self.demo_curriculum_step < 0] = True
                # self.demo_curriculum_step = torch.clamp(self.demo_curriculum_step, 0)

                # if can_advance.any():
                #     # update the probability distribution for trajectory sampling
                #     self._traj_index_density = torch.divide(self.demo_curriculum_step, self.demo_horizon)
                #     self._traj_index_density[self.demo_curriculum_step == 0] = 1e-6
                #     self._traj_index_density = (self._traj_index_density / self._traj_index_density.sum()).cpu().numpy()

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
                x_start_steps_density_list = np.array([0.5, 0.25, 0.125, 0.125 / 2, 0.125 / 2])
                sampled_offsets = self.base_env._episode_rng.choice(np.arange(0, len(x_start_steps_density_list)), size=(b, ), replace=True, p=x_start_steps_density_list)
                sampled_offsets = torch.from_numpy(sampled_offsets).to(self.base_env.device)
            elif self.reverse_curriculum_sampler == "point":
                sampled_offsets = self.demo_curriculum_step[reset_traj_indexes] * 0
            elif self.reverse_curriculum_sampler == "bigeo":
                pass

            x_start_steps = sampled_offsets
            for i, traj_idx in enumerate(reset_traj_indexes):
                x_start_steps[i] += self.demo_metadata[traj_idx.item()].start_step
            x_start_steps = torch.clamp(x_start_steps, torch.zeros((b, ), device=self.base_env.device), self.demo_horizon[reset_traj_indexes] - 1).int()
            self.dynamic_max_episode_steps[env_idx] = self.max_steps_min + ((self.demo_horizon[reset_traj_indexes] - x_start_steps) // self.demo_horizon_to_max_steps_ratio).int()
            self.sampled_start_steps[env_idx] = x_start_steps
            self.base_env.set_state(self.env_states[reset_traj_indexes, x_start_steps], env_idx)
        obs = self.base_env.get_obs()
        return obs, {"reconfigure": False}


@dataclass
class ForwardCurriculumConfig:
    forward_curriculum: str = "success_once_score"
    staleness_coef: float =  0.1
    staleness_temperature: float =  0.1
    staleness_transform: str = "rankmin"
    score_transform: str = "rankmin"
    score_temperature: float = 0.1