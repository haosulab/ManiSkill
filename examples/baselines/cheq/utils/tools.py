import random
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any, NamedTuple, Optional

import numpy as np
from sac.sac import ReplayBuffer
from gymnasium import spaces
import gymnasium as gym
import torch
from stable_baselines3.common.vec_env import VecNormalize



class BernoulliMaskReplayBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    masks: torch.Tensor


def seed_experiment(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def convert_to_tensor(data, device):

    if isinstance(data, torch.Tensor):
        data.to(device=device)
        return data
    else:
        tensor = torch.tensor(data, device=device)
        return tensor

def get_kwargs(dict):
    """
    Gets the kwargs provided in the config files
    Args:
        dict:

    Returns: kwargs

    """
    return {k: v for k, v in dict.items() if k != 'type'}


def inject_weight_into_state(state, weight):
    if len(state.shape) == 1:
        weight_array = np.array([weight], dtype=np.float32)
        array = np.append(state.copy(), weight_array)
    else:
        weight_array = np.array([[weight]], dtype=np.float32)
        array = np.append(state.copy(), weight_array, axis=1)

    return array


def compute_clipped_cv(std, mean):
    clipped_cv = np.where(np.abs(mean)>1, std/np.abs(mean), std)
    return clipped_cv


def compute_distance_between_vectors(tensor1, tensor2):
    assert tensor1.shape == tensor2.shape
    with torch.no_grad():
        return (tensor1 - tensor2).pow(2).sum(dim=-1).sqrt()


def make_env(env_id, seed):
    """
    Returns a thunk that creates and initializes a gym environment with the given ID and seed
    Args:
        env_id: string identifying the gym environment to create
        seed: integer specifying the random seed to use for the environment
    Returns:
        callable thunk that creates and returns a gym environment with a seeded initial state, action space, and observation spaces
    """

    def thunk():
        env = gym.make(env_id)
        # env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


class SimpleRingBuffer:

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.cur_index = 0
        self.buffer = []

    def add_element(self, element):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(element)
        else:
            self.buffer[self.cur_index % self.buffer_size] = element
            self.cur_index += 1

    def get_buffer_mean(self):
        return np.mean(self.buffer)

    def clear_buffer(self):
        self.buffer = []
        self.cur_index = 0


class WeightScheduler(ABC):

    def __init__(self, start_weight):
        self.current_weight = start_weight

    @abstractmethod
    def adapt_weight(self, uncertainty, step, force=False):
        pass

    @abstractmethod
    def episode_weight_reset(self):
        pass

    def get_weight(self):
        return self.current_weight


class MinResetWeightScheduler(WeightScheduler, ABC):
    def __init__(self, lambda_min):
        super().__init__(lambda_min)
        self.lambda_min = lambda_min

    def episode_weight_reset(self):
        self.current_weight = self.lambda_min


class FixedWeightScheduler(WeightScheduler):

    def __init__(self, weight):
        super().__init__(weight)

    def episode_weight_reset(self):
        pass

    def adapt_weight(self, uncertainty, step, force=False):
        pass


class LinearTimeWeightScheduler(WeightScheduler):
    def __init__(self, start_weight, end_weight, start_step, end_step):
        self.end_step = end_step
        self.start_step = start_step
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.current_weight = start_weight

    def adapt_weight(self, uncertainty, step, force=False):
        m = (self.end_weight - self.start_weight)/(self.end_step - self.start_step)
        self.current_weight = m * (step - self.start_step) + self.start_weight
        self.current_weight = np.clip(self.current_weight, self.start_weight, self.end_weight)

    def episode_weight_reset(self):
        pass

    def get_weight(self):
        return self.current_weight


class CoreExponentialWeightScheduler(MinResetWeightScheduler):

    def __init__(self, factor_a, factor_c, t_start, lambda_warmup=None, lambda_warmup_max=None):
        super().__init__(1/(1 + factor_a))
        self.mapper = InverseExponentialMapper(factor_a, factor_c)
        self.t_start = t_start

        self.lambda_min = lambda_warmup_max if lambda_warmup_max is not None else 1/(1 + factor_a)
        self.lambda_max = 1

        if lambda_warmup is None:
            self.lambda_warmup = self.lambda_min
        else:
            self.lambda_warmup = lambda_warmup

        if lambda_warmup_max is None:
            self.lambda_warmup_max = self.lambda_warmup
        else:
            self.lambda_warmup_max = lambda_warmup_max

    def adapt_weight(self, uncertainty, step, force=False):

            if force or step > self.t_start:
                self.current_weight = self.mapper.get_mapped_value(uncertainty)
            else:
                self.current_weight = np.random.uniform(self.lambda_warmup, self.lambda_warmup_max)

    def episode_weight_reset(self):
        self.current_weight = self.lambda_warmup


class MovingAVGLinearWeightScheduler(MinResetWeightScheduler):
    def __init__(self, lambda_min, t_start, u_high, u_low, window_size, lambda_max=1.0, lambda_warmup_max=None, t_full=None,
                 behaviour_cloning_steps=None):
        super().__init__(lambda_min)
        self.smoother = MovingAverageSmoother(window_size)
        self.mapper = LinearMapper(lambda_min=lambda_min, lamnda_max=lambda_max, u_high=u_high, u_low=u_low)
        self.t_start = t_start
        if t_full is None:
            self.t_full = t_start
        else:
            self.t_full = t_full
        if lambda_warmup_max is None:
            self.lambda_warmup_max = lambda_min
        else:
            self.lambda_warmup_max = lambda_warmup_max

        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        if behaviour_cloning_steps is None:
            self.behaviour_cloning_steps = 0
        else:
            self.behaviour_cloning_steps = behaviour_cloning_steps

        assert self.behaviour_cloning_steps <= self.t_start <= self.t_full, 'The thresholds are inconsistent.'

    def adapt_weight(self, uncertainty, step, force=False):
        self.smoother.add_value(uncertainty)
        uncertainty_level = self.smoother.get_smoothed_value()

        if force:
            self.current_weight = self.mapper.get_mapped_value(uncertainty_level)
        elif step < self.behaviour_cloning_steps:
            # randomly sample weight between lambda_min and lambda_max
            self.current_weight = np.random.uniform(self.lambda_min, self.lambda_max)
        elif step > self.t_start:
            self.current_weight = self.mapper.get_mapped_value(uncertainty_level)
            if step < self.t_full:  # limiting the maximal lambda
                step_position = (step - self.t_start) / (self.t_full - self.t_start)
                cur_max_weight = self.lambda_warmup_max + step_position * (self.lambda_max - self.lambda_warmup_max)
                self.current_weight = min(cur_max_weight, self.current_weight)
        else:
            self.current_weight = np.random.uniform(self.lambda_min, self.lambda_warmup_max)

    def episode_weight_reset(self):
        self.current_weight = self.lambda_min
        self.smoother.clear_history()


class Smoother(ABC):

    @abstractmethod
    def add_value(self, next_value):
        pass

    @abstractmethod
    def get_smoothed_value(self):
        pass

    @abstractmethod
    def clear_history(self):
        pass


class MovingAverageSmoother(Smoother):

    def __init__(self, window_size):
        self.window_size = window_size
        self.buffer = SimpleRingBuffer(window_size)

    def add_value(self, next_value):
        self.buffer.add_element(next_value)

    def get_smoothed_value(self,):
        return self.buffer.get_buffer_mean()

    def clear_history(self):
        self.buffer.clear_buffer()


class Mapper(ABC):

    @abstractmethod
    def get_mapped_value(self, value):
        pass


class LinearMapper(Mapper):

    def __init__(self, lambda_min, lamnda_max, u_low, u_high):
        self.lambda_min = lambda_min
        self.lambda_max = lamnda_max
        self.u_low = u_low
        self.u_high = u_high

    def get_mapped_value(self, value):
        m = (self.lambda_max - self.lambda_min) / (self.u_low - self.u_high)
        mapped_val = m * (value - self.u_low) + self.lambda_max

        # clipping the weight to [min, max]
        mapped_val = min(max(mapped_val, self.lambda_min), self.lambda_max)

        return mapped_val


class InverseExponentialMapper(Mapper):

    def __init__(self, factor_a, factor_c):
        self.factor_a = factor_a
        self.factor_c = factor_c

    def get_mapped_value(self, value):
        x = self.factor_a*(1-np.exp(-self.factor_c * value))
        return 1/(1+x)


class UncertaintyEvaluator(ABC):

    @abstractmethod
    def get_uncertainty(self, state, action, state_next, reward):
        """

        Args:
            state: tensor of RL state
            action: tensor of RL action
            state_next: tensor of next RL state
            reward: scalar value of reward of the timestep leading from state to state_next

        Returns:
        scalar uncertainty metric
        """
        pass


class DummyUncertaintyEvaluator(UncertaintyEvaluator):

    def get_uncertainty(self, state, action, state_next, reward):
        return 42


class QEnsembleSTDUncertaintyEvaluator(UncertaintyEvaluator):
    def __init__(self, agent, device='cpu'):
        self.agent = agent
        self.device = device

    def get_uncertainty(self, state, action, state_next, reward):
        with torch.no_grad():
            _, std = self.agent.get_ensemble_std(convert_to_tensor(state, device=self.device),
                                                 convert_to_tensor(action, device=self.device))
        return std.item()


class TargetTDErrorUncertaintyEvaluator(UncertaintyEvaluator):
    """
    Uncertainty Evaluation a la CORE
    """
    def __init__(self, agent, device='cpu', gamma=0.99):
        self.agent = agent
        self.gamma = gamma
        self.device = device

    def get_uncertainty(self, state, action, state_next, reward):
        with torch.no_grad():

            act_b = self.agent.get_action(convert_to_tensor(state, self.device), greedy=True)
            base_q = self.agent.get_target_q1_value(convert_to_tensor(state, self.device), act_b)
            act_t = self.agent.get_action(convert_to_tensor(state_next, device=self.device), greedy=True)
            target_q = self.agent.get_target_q1_value(convert_to_tensor(state_next, device=self.device), act_t)

        td_error = reward + self.gamma * target_q.item() - base_q.item()

        return abs(td_error)


class BernoulliMaskReplayBuffer(ReplayBuffer):

    def __init__(
            self,
            buffer_size: int,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            mask_size : int,
            p_masking: float,
            device: Union[torch.device, str] = "cpu",
            n_envs: int = 1,
            optimize_memory_usage: bool = False,
            handle_timeout_termination: bool = True,
    ):
        super(BernoulliMaskReplayBuffer, self).__init__(buffer_size=buffer_size,
                                           observation_space=observation_space,
                                           action_space=action_space,
                                           device=device,
                                           n_envs=n_envs,
                                           optimize_memory_usage=optimize_memory_usage,
                                           handle_timeout_termination=handle_timeout_termination)

        self.mask_size = mask_size
        self.bernoulli_mask = np.zeros((self.buffer_size, self.n_envs, self.mask_size), dtype=np.float32)

        self.p_masking = p_masking

    def add(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],
    ) -> None:

        super(BernoulliMaskReplayBuffer, self).add(obs, next_obs, action, reward, done, infos)

        # sample bernoulli mask
        mask = np.random.choice([1., 0.], p=[self.p_masking, 1-self.p_masking], size=(self.n_envs, self.mask_size)).astype(np.float32)
        self.bernoulli_mask[self.pos-1] = mask

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> BernoulliMaskReplayBufferSamples:
        if not self.optimize_memory_usage:
            return super(ReplayBuffer, self).sample(batch_size=batch_size, env=env)
            # Do not sample the element with index `self.pos` as the transitions is invalid
            # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)


        return self._get_samples(batch_inds, env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> BernoulliMaskReplayBufferSamples:
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, 0, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, 0, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, 0, :], env),
            self.actions[batch_inds, 0, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            self.dones[batch_inds] * (1 - self.timeouts[batch_inds]),
            self._normalize_reward(self.rewards[batch_inds], env),
            self.bernoulli_mask[batch_inds, 0, :]

        )
        return BernoulliMaskReplayBufferSamples(*tuple(map(self.to_torch, data)))





