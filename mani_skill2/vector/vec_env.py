"""Vectorized environment for ManiSkill2.

See also:
    https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/subproc_vec_env.py
"""


import multiprocessing as mp
import os
from functools import partial
from multiprocessing.connection import Connection
from typing import Callable, List, Optional, Sequence, Union

import gym
import numpy as np
import sapien.core as sapien
import torch
from gym import spaces
from gym.vector.utils.shared_memory import *

from mani_skill2 import logger
from mani_skill2.envs.sapien_env import BaseEnv


def _worker(
    rank: int,
    remote: Connection,
    parent_remote: Connection,
    env_fn: Callable[..., BaseEnv],
    shared_memory=None,
):
    # NOTE(jigu): Set environment variables for ManiSkill2
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    parent_remote.close()

    try:
        env = env_fn()
        obs_space = env.observation_space
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, done, info = env.step(data)
                if shared_memory is not None:
                    write_to_shared_memory(rank, obs, shared_memory, obs_space)
                    obs = None
                remote.send((obs, reward, done, info))
            elif cmd == "reset":
                obs = env.reset()
                if shared_memory is not None:
                    write_to_shared_memory(rank, obs, shared_memory, obs_space)
                    obs = None
                remote.send(obs)
            elif cmd == "close":
                remote.close()
                break
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == "handshake":
                remote.send(None)
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
    except KeyboardInterrupt:
        logger.info("Worker KeyboardInterrupt")
    except EOFError:
        logger.info("Worker EOF")
    except Exception as err:
        logger.error(err, exec_info=1)
    finally:
        env.close()


class VecEnv:
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: Environments to run in subprocesses
    :param start_method: method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(
        self,
        env_fns: List[Callable[[], BaseEnv]],
        start_method: Optional[str] = None,
        shared_memory=False,
        server_address=None,
    ):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        # Acquire observation space to construct buffer
        # NOTE(jigu): Use a separate process to avoid creating sapien resources in the main process
        remote, work_remote = ctx.Pipe()
        args = (0, work_remote, remote, env_fns[0], None)
        process = ctx.Process(target=_worker, args=args, daemon=True)
        process.start()
        work_remote.close()
        remote.send(("get_attr", "observation_space"))
        self.observation_space = remote.recv()
        remote.send(("get_attr", "action_space"))
        self.action_space = remote.recv()
        remote.send(("close", None))
        remote.close()
        process.join()

        # Allocate shared memory
        self.num_envs = n_envs
        self.shared_memory = shared_memory
        self.vec_observation_space = stack_observation_space(
            self.observation_space, n_envs
        )
        if self.shared_memory:
            shm = create_shared_memory(self.observation_space, n_envs)
            self._obs_buffer = read_from_shared_memory(
                shm, self.observation_space, n=n_envs
            )
        else:
            shm = None
            # self._obs_buffer = create_np_buffer(self.vec_observation_space)
            self._obs_buffer = None

        # Wrap env_fn if using sapien.RenderServer
        self.server_address = server_address
        if self.server_address is not None:
            # TODO(jigu): how to deal with initialization arguments?
            self.server = sapien.RenderServer()
            self.server.start(self.server_address)
            for i, env_fn in enumerate(env_fns):
                client_kwargs = {"address": self.server_address, "process_index": i}
                env_fns[i] = partial(
                    env_fn, renderer="client", client_kwargs=client_kwargs
                )

        # Initialize workers
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for rank in range(n_envs):
            work_remote = self.work_remotes[rank]
            remote = self.remotes[rank]
            env_fn = env_fns[rank]
            args = (rank, work_remote, remote, env_fn, shm)
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(
                target=_worker, args=args, daemon=True
            )  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()

        if self.server_address is not None:
            self.handshake()  # To make sure environments are all initialized
            image_obs_space = self.observation_space["image"]
            texture_names = set()
            for name, subspace in image_obs_space.spaces.items():
                texture_names.update(subspace.spaces.keys())
            self.texture_names = tuple(texture_names)
            # List of [n_envs, n_cams, H, W, C]
            self._obs_tensors = self.server.auto_allocate_torch_tensors(
                self.texture_names
            )

    def handshake(self):
        for remote in self.remotes:
            remote.send(("handshake", None))
        for remote in self.remotes:
            remote.recv()

    def step_async(self, actions: np.ndarray) -> None:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        vec_obs = self.vectorize_obs(obs)
        return vec_obs, rews, dones, infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        if seed is None:
            seed = np.random.randint(0, 2**32)
        for idx, remote in enumerate(self.remotes):
            remote.send(("seed", seed + idx))
        return [remote.recv() for remote in self.remotes]

    def reset_async(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        self.waiting = True

    def reset_wait(self):
        obs = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return self.vectorize_obs(obs)

    def vectorize_obs(self, obs):
        if self.server_address is not None:
            return self.vectorize_obs_and_image(obs)
        elif self.shared_memory:
            return self._obs_buffer
        else:
            return stack_obs(obs, self.observation_space, self._obs_buffer)
            # return obs

    def vectorize_obs_and_image(self, obs):
        self.server.wait_all()
        vec_obs = stack_obs(obs, self.observation_space, self._obs_buffer, strict=False)

        # Add image observations from server
        image_obs = {}
        cam_idx = 0
        image_obs_space = self.observation_space.spaces["image"]
        for cam_name, cam_obs_space in image_obs_space.spaces.items():
            image_obs[cam_name] = {}
            for tex_name in cam_obs_space:
                tex_idx = self.texture_names.index(tex_name)
                # TODO(jigu): deal with non-equal shape
                image_obs[cam_name][tex_name] = self._obs_tensors[tex_idx][:, cam_idx]
            cam_idx += 1
        vec_obs["image"] = image_obs
        return vec_obs

    def reset(self):
        self.reset_async()
        return self.reset_wait()

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_attr(self, attr_name: str, indices=None) -> List:
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices=None,
        **method_kwargs,
    ) -> List:
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def _get_indices(self, indices) -> List[int]:
        """
        Convert a flexibly-typed reference to environment indices to an implied list of indices.

        :param indices: refers to indices of envs.
        :return: the implied list of indices.
        """
        if indices is None:
            indices = list(range(self.num_envs))
        elif isinstance(indices, int):
            indices = [indices]
        return indices

    def _get_target_remotes(self, indices) -> List[Connection]:
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: refers to indices of envs.
        :return: Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]

    def __repr__(self):
        return "{}({})".format(
            self.__class__.__name__, self.env_method("__repr__", indices=0)[0]
        )


def stack_observation_space(space: spaces.Space, n_envs: int):
    if isinstance(space, spaces.Dict):
        sub_spaces = [
            (key, stack_observation_space(subspace, n_envs))
            for key, subspace in space.spaces.items()
        ]
        return spaces.Dict(sub_spaces)
    elif isinstance(space, spaces.Box):
        shape = (n_envs,) + space.shape
        low = np.broadcast_to(space.low, shape)
        high = np.broadcast_to(space.high, shape)
        return spaces.Box(low=low, high=high, shape=shape, dtype=space.dtype)
    else:
        raise NotImplementedError(
            "Unsupported observation space: {}".format(type(space))
        )


def create_np_buffer(space: spaces.Space):
    if isinstance(space, spaces.Dict):
        return {
            key: create_np_buffer(subspace) for key, subspace in space.spaces.items()
        }
    elif isinstance(space, spaces.Box):
        return np.zeros(space.shape, dtype=space.dtype)
    else:
        raise NotImplementedError(
            "Unsupported observation space: {}".format(type(space))
        )


def stack_obs(
    obs: Sequence, space: spaces.Space, buffer: Optional[np.ndarray] = None, strict=True
):
    if isinstance(space, spaces.Dict):
        ret = {}
        for key in space:
            if key not in obs[0]:
                if strict:
                    raise KeyError(f"Key {key} not found in observation")
                else:
                    continue
            _obs = [o[key] for o in obs]
            _buffer = None if buffer is None else buffer[key]
            ret[key] = stack_obs(_obs, space[key], buffer=_buffer, strict=strict)
        return ret
    elif isinstance(space, spaces.Box):
        return np.stack(obs, out=buffer)
    else:
        raise NotImplementedError(type(space))


class VecEnvWrapper(VecEnv):
    def __init__(self, venv: VecEnv):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space

    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    def step_wait(self):
        return self.venv.step_wait()

    def reset_async(self):
        self.venv.reset_async()

    def reset_wait(self):
        return self.venv.reset_wait()

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        return self.venv.seed(seed)

    def close(self) -> None:
        return self.venv.close()

    def get_attr(self, attr_name: str, indices=None) -> List:
        return self.venv.get_attr(attr_name, indices)

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        return self.venv.set_attr(attr_name, value, indices)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices=None,
        **method_kwargs,
    ) -> List:
        return self.venv.env_method(
            method_name, *method_args, indices=indices, **method_kwargs
        )

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            return getattr(self.venv, name)


class VecEnvObservationWrapper(VecEnvWrapper):
    def reset_wait(self, **kwargs):
        observation = self.venv.reset_wait(**kwargs)
        return self.observation(observation)

    def step_wait(self):
        observation, reward, done, info = self.venv.step_wait()
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        raise NotImplementedError
