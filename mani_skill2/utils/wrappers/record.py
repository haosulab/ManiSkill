import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import gymnasium as gym
import h5py
import numpy as np
import sapien.physx as physx

from mani_skill2 import get_commit_info
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.common import (
    extract_scalars_from_info,
    find_max_episode_steps_value,
)
from mani_skill2.utils.io_utils import dump_json
from mani_skill2.utils.sapien_utils import batch, to_numpy
from mani_skill2.utils.structs.types import Array
from mani_skill2.utils.visualization.misc import (
    images_to_video,
    put_info_on_image,
    tile_images,
)

# NOTE (stao): The code for record.py is quite messy and perhaps confusing as it is trying to support both recording on CPU and GPU seamlessly
# and handle partial resets. It works but can be claned up a lot.


def append_dict_array(
    x1: Union[dict, Sequence, Array], x2: Union[dict, Sequence, Array]
):
    """Append `x2` in front of `x1` and returns the result. Tries to do this in place if possible.
    Assumes both `x1, x2` have the same dictionary structure if they are dictionaries.
    They may also both be lists/sequences in which case this is just appending like normal"""
    if isinstance(x1, np.ndarray):
        if len(x1.shape) > len(x2.shape):
            # if different dims, check if extra dim is just a 1 due to single env in batch mode and if so, add it to x2.
            if x1.shape[1] == 1:
                x2 = x2[:, None, :]
        return np.concatenate([x1, x2])
    elif isinstance(x1, list):
        return x1 + x2
    elif isinstance(x1, dict):
        for k in x1.keys():
            assert k in x2, "dct and append_dct need to have the same dictionary layout"
            x1[k] = append_dict_array(x1[k], x2[k])
    return x1


def slice_dict_array(x1, slice: slice):
    """Slices every array in x1 with slice and returns result. Tries to do this in place if possible"""
    if isinstance(x1, np.ndarray) or isinstance(x1, list):
        return x1[slice]
    elif isinstance(x1, dict):
        for k in x1.keys():
            x1[k] = slice_dict_array(x1[k], slice)
    return x1


def parse_env_info(env: gym.Env):
    # spec can be None if not initialized from gymnasium.make
    env = env.unwrapped
    if env.spec is None:
        return None
    if hasattr(env.spec, "_kwargs"):
        # gym<=0.21
        env_kwargs = env.spec._kwargs
    else:
        # gym>=0.22
        env_kwargs = env.spec.kwargs
    return dict(
        env_id=env.spec.id,
        env_kwargs=env_kwargs,
    )


def temp_deep_print_shapes(x, prefix=""):
    if isinstance(x, dict):
        for k in x:
            temp_deep_print_shapes(x[k], prefix=prefix + "/" + k)
    else:
        print(prefix, x.shape)


def clean_trajectories(h5_file: h5py.File, json_dict: dict, prune_empty_action=True):
    """Clean trajectories by renaming and pruning trajectories in place.

    After cleanup, trajectory names are consecutive integers (traj_0, traj_1, ...),
    and trajectories with empty action are pruned.

    Args:
        h5_file: raw h5 file
        json_dict: raw JSON dict
        prune_empty_action: whether to prune trajectories with empty action
    """
    json_episodes = json_dict["episodes"]
    assert len(h5_file) == len(json_episodes)

    # Assumes each trajectory is named "traj_{i}"
    prefix_length = len("traj_")
    ep_ids = sorted([int(x[prefix_length:]) for x in h5_file.keys()])

    new_json_episodes = []
    new_ep_id = 0

    for i, ep_id in enumerate(ep_ids):
        traj_id = f"traj_{ep_id}"
        ep = json_episodes[i]
        assert ep["episode_id"] == ep_id
        new_traj_id = f"traj_{new_ep_id}"

        if prune_empty_action and ep["elapsed_steps"] == 0:
            del h5_file[traj_id]
            continue

        if new_traj_id != traj_id:
            ep["episode_id"] = new_ep_id
            h5_file[new_traj_id] = h5_file[traj_id]
            del h5_file[traj_id]

        new_json_episodes.append(ep)
        new_ep_id += 1

    json_dict["episodes"] = new_json_episodes


@dataclass
class Step:
    state: np.ndarray
    observation: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    done: np.ndarray
    env_episode_ptr: np.ndarray
    """points to index in above data arrays where current episode started (any data before should already be flushed)"""

    success: np.ndarray = None
    fail: np.ndarray = None


class RecordEpisode(gym.Wrapper):
    """Record trajectories or videos for episodes. You generally should always apply this wrapper last, particularly if you include
    observation wrappers which modify the returned observations. The only wrappers that may go after this one is any of the vector env
    interface wrappers that map the maniskill env to a e.g. gym vector env interface.

    Trajectory data is saved with two files, the actual data in a .h5 file via H5py and metadata in a JSON file of the same basename.

    Each JSON file contains:

    - `env_info` (Dict): environment information, which can be used to initialize the environment
    - `env_id` (str): environment id
    - `max_episode_steps` (int)
    - `env_kwargs` (Dict): keyword arguments to initialize the environment. **Essential to recreate the environment.**
    - `episodes` (List[Dict]): episode information

    The episode information (the element of `episodes`) includes:

    - `episode_id` (int): a unique id to index the episode
    - `reset_kwargs` (Dict): keyword arguments to reset the environment. **Essential to reproduce the trajectory.**
    - `control_mode` (str): control mode used for the episode.
    - `elapsed_steps` (int): trajectory length
    - `info` (Dict): information at the end of the episode.

    With just the meta data, you can reproduce the environment the same way it was created when the trajectories were collected as so:

    ```python
    env = gym.make(env_info["env_id"], **env_info["env_kwargs"])
    episode = env_info["episodes"][0] # picks the first
    env.reset(**episode["reset_kwargs"])
    ```

    Each HDF5 demonstration dataset consists of multiple trajectories. The key of each trajectory is `traj_{episode_id}`, e.g., `traj_0`.

    Each trajectory is an `h5py.Group`, which contains:

    - actions: [T, A], `np.float32`. `T` is the number of transitions.
    - success: [T], `np.bool_`. It indicates whether the task is successful at each time step.
    - env_states: [T+1, D], `np.float32`. Environment states. It can be used to set the environment to a certain state, e.g., `env.set_state(env_states[i])`. However, it may not be enough to reproduce the trajectory.
    - obs (optional): observations. If the observation is a `dict`, the value will be stored in `obs/{key}`. The convention is applied recursively for nested dict.


    Args:
        env: gym.Env
        output_dir: output directory
        save_trajectory: whether to save trajectory
        trajectory_name: name of trajectory file (.h5). Use timestamp if not provided.
        save_video: whether to save video
        info_on_video: whether to write data about reward, action, and data in the info object to the video. The first video frame is generally the result
            of the first env.reset() (visualizing the first observation). Text is written on frames after that, showing the action taken to get to that
            environment state and reward.
        save_on_reset: whether to save the previous trajectory (and video of it if `save_video` is True) automatically when resetting.
            Not that for environments simulated on the GPU (to leverage fast parallel rendering) you must
            set `max_steps_per_video` to a fixed number so that every `max_steps_per_video` steps a video is saved. This is
            required as there may be partial environment resets which makes it ambiguous about how to save/cut videos.
        max_steps_per_video: how many steps can be recorded into a single video before flushing the video. If None this is not used. A internal step counter is maintained to do this.
            If the video is flushed at any point, the step counter is reset to 0.
        clean_on_close: whether to rename and prune trajectories when closed.
            See `clean_trajectories` for details.
        video_fps (int): The FPS of the video to generate if save_video is True
    """

    def __init__(
        self,
        env,
        output_dir,
        save_trajectory=True,
        trajectory_name=None,
        save_video=True,
        info_on_video=False,
        save_on_reset=True,
        max_steps_per_video=None,
        clean_on_close=True,
        record_reward=True,
        video_fps=20,
    ):
        super().__init__(env)

        self.output_dir = Path(output_dir)
        if save_trajectory or save_video:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_fps = video_fps
        self._episode_id = -1
        self._video_id = -1
        self._video_steps = 0
        self._closed = False

        self._trajectory_buffer: Step = None

        self.max_steps_per_video = max_steps_per_video
        self.max_episode_steps = find_max_episode_steps_value(env)

        self.save_on_reset = save_on_reset
        self.save_trajectory = save_trajectory
        if self._base_env.num_envs > 1 and save_video:
            assert (
                max_steps_per_video is not None
            ), "On GPU parallelized environments, \
                there must be a given max steps per video value in order to flush videos in order \
                to avoid issues caused by partial resets. If your environment does not do partial \
                resets you may set max_steps_per_video equal to the max_episode_steps"
        self.clean_on_close = clean_on_close
        self.record_reward = record_reward
        if self.save_trajectory:
            if not trajectory_name:
                trajectory_name = time.strftime("%Y%m%d_%H%M%S")

            self._h5_file = h5py.File(self.output_dir / f"{trajectory_name}.h5", "w")

            # Use a separate json to store non-array data
            self._json_path = self._h5_file.filename.replace(".h5", ".json")
            self._json_data = dict(
                env_info=parse_env_info(self.env),
                commit_info=get_commit_info(),
                episodes=[],
            )
            self._json_data["env_info"]["max_episode_steps"] = self.max_episode_steps

        self.save_video = save_video
        self.info_on_video = info_on_video
        self._render_images = []
        if info_on_video and physx.is_gpu_enabled():
            raise ValueError(
                "Cannot turn info_on_video=True when using GPU simulation as the text would be too small"
            )
        self.video_nrows = int(np.sqrt(self.unwrapped.num_envs))

    @property
    def num_envs(self):
        return self._base_env.num_envs

    @property
    def _base_env(self) -> BaseEnv:
        return self.env.unwrapped

    def capture_image(self):
        img = self.env.render()
        img = to_numpy(img)
        if len(img.shape) > 3:
            img = tile_images(img, nrows=self.video_nrows)
        return img

    def reset(
        self,
        *args,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[dict] = dict(),
        **kwargs,
    ):
        if self.save_on_reset and self._trajectory_buffer is not None:
            if self.save_video and self.num_envs == 1:
                self.flush_video()
            # if doing a full reset then we flush all trajectories including incompleted ones
            if "env_idx" not in options:
                self.flush_trajectory(env_idxs_to_flush=np.arange(self.num_envs))
            else:
                self.flush_trajectory(env_idxs_to_flush=to_numpy(options["env_idx"]))

        self.last_reset_kwargs = copy.deepcopy(
            dict(seed=seed, options=options, **kwargs)
        )
        obs, info = super().reset(*args, seed=seed, options=options, **kwargs)

        if self.save_trajectory:
            state_dict = self._base_env.get_state_dict()
            action = batch(self.action_space.sample())
            first_step = Step(
                state=to_numpy(batch(state_dict)),
                observation=to_numpy(batch(obs)),
                # note first reward/action etc. are ignored when saving trajectories to disk
                action=action,
                reward=np.zeros(
                    (
                        1,
                        self.num_envs,
                    ),
                    dtype=float,
                ),
                # terminated and truncated are fixed to be True at the start to indicate the start of an episode.
                # an episode is done when one of these is True otherwise the trajectory is incomplete / a partial episode
                terminated=np.ones((1, self.num_envs), dtype=bool),
                truncated=np.ones((1, self.num_envs), dtype=bool),
                done=np.ones((1, self.num_envs), dtype=bool),
                success=np.zeros((1, self.num_envs), dtype=bool),
                fail=np.zeros((1, self.num_envs), dtype=bool),
                env_episode_ptr=np.zeros((self.num_envs,), dtype=int),
            )
            if self.num_envs == 1:
                first_step.observation = batch(first_step.observation)
                first_step.action = batch(first_step.action)
            env_idx = np.arange(self.num_envs)
            if "env_idx" in options:
                env_idx = to_numpy(options["env_idx"])
            if self._trajectory_buffer is None:
                # Initialize trajectory buffer on the first episode based on given observation (which should be generated after all wrappers)
                self._trajectory_buffer = first_step
            else:

                def recursive_replace(x, y):
                    if isinstance(x, np.ndarray):
                        x[-1, env_idx] = y[-1, env_idx]
                    else:
                        for k in x.keys():
                            recursive_replace(x[k], y[k])

                recursive_replace(self._trajectory_buffer.state, first_step.state)
                recursive_replace(
                    self._trajectory_buffer.observation, first_step.observation
                )
                recursive_replace(self._trajectory_buffer.action, first_step.action)
                recursive_replace(self._trajectory_buffer.reward, first_step.reward)
                recursive_replace(
                    self._trajectory_buffer.terminated, first_step.terminated
                )
                recursive_replace(
                    self._trajectory_buffer.truncated, first_step.truncated
                )
                recursive_replace(self._trajectory_buffer.done, first_step.done)
                if self._trajectory_buffer.success is not None:
                    recursive_replace(
                        self._trajectory_buffer.success, first_step.success
                    )
                if self._trajectory_buffer.fail is not None:
                    recursive_replace(self._trajectory_buffer.fail, first_step.fail)

        return obs, info

    def step(self, action):
        if self.save_video and self._video_steps == 0:
            # save the first frame of the video here (s_0) instead of inside reset as user
            # may call env.reset(...) multiple times but we want to ignore empty trajectories
            self._render_images.append(self.capture_image())
        obs, rew, terminated, truncated, info = super().step(action)

        if self.save_trajectory:
            if (
                isinstance(truncated, bool)
                and self.num_envs > 1
                and self.max_episode_steps is not None
            ):
                # this fixes the issue where gymnasium applies a non-batched timelimit wrapper
                truncated = self._base_env.elapsed_steps >= self.max_episode_steps
            state_dict = self._base_env.get_state_dict()
            self._trajectory_buffer.state = append_dict_array(
                self._trajectory_buffer.state, to_numpy(batch(state_dict))
            )
            self._trajectory_buffer.observation = append_dict_array(
                self._trajectory_buffer.observation, to_numpy(batch(obs))
            )

            self._trajectory_buffer.action = append_dict_array(
                self._trajectory_buffer.action, to_numpy(batch(action))
            )
            self._trajectory_buffer.reward = append_dict_array(
                self._trajectory_buffer.reward, to_numpy(batch(rew))
            )
            self._trajectory_buffer.terminated = append_dict_array(
                self._trajectory_buffer.terminated, to_numpy(batch(terminated))
            )
            self._trajectory_buffer.truncated = append_dict_array(
                self._trajectory_buffer.truncated, to_numpy(batch(truncated))
            )
            done = terminated | truncated
            self._trajectory_buffer.done = append_dict_array(
                self._trajectory_buffer.done, to_numpy(batch(done))
            )
            if "success" in info:
                self._trajectory_buffer.success = append_dict_array(
                    self._trajectory_buffer.success, to_numpy(batch(info["success"]))
                )
            else:
                self._trajectory_buffer.success = None
            if "fail" in info:
                self._trajectory_buffer.fail = append_dict_array(
                    self._trajectory_buffer.fail, to_numpy(batch(info["fail"]))
                )
            else:
                self._trajectory_buffer.fail = None
            self._last_info = to_numpy(info)

        if self.save_video:
            self._video_steps += 1
            image = self.capture_image()

            if self.info_on_video:
                scalar_info = extract_scalars_from_info(info)
                extra_texts = [
                    f"reward: {rew:.3f}",
                    "action: {}".format(",".join([f"{x:.2f}" for x in action])),
                ]
                image = put_info_on_image(image, scalar_info, extras=extra_texts)

            self._render_images.append(image)
            if (
                self.max_steps_per_video is not None
                and self._video_steps >= self.max_steps_per_video
            ):
                self.flush_video()

        return obs, rew, terminated, truncated, info

    def flush_trajectory(
        self,
        verbose=False,
        ignore_empty_transition=False,
        env_idxs_to_flush=[],
    ):
        flush_count = 0
        for env_idx in env_idxs_to_flush:
            start_ptr = self._trajectory_buffer.env_episode_ptr[env_idx]
            end_ptr = len(self._trajectory_buffer.done)
            if ignore_empty_transition and end_ptr - start_ptr <= 1:
                continue
            self._episode_id += 1

            traj_id = "traj_{}".format(self._episode_id)
            group = self._h5_file.create_group(traj_id, track_order=True)

            def recursive_add_to_h5py(group: h5py.Group, data: dict, key):
                """simple recursive data insertion for nested data structures into h5py, optimizing for visual data as well"""
                if isinstance(data, dict):
                    subgrp = group.create_group(key, track_order=True)
                    for k in data.keys():
                        recursive_add_to_h5py(subgrp, data[k], k)
                else:
                    if key == "rgb":
                        # NOTE(jigu): It is more efficient to use gzip than png for a sequence of images.
                        group.create_dataset(
                            "rgb",
                            data=data[start_ptr:end_ptr, env_idx],
                            dtype=data.dtype,
                            compression="gzip",
                            compression_opts=5,
                        )
                    elif key == "depth":
                        # NOTE (stao): By default now cameras in ManiSkill return depth values of type uint16 for numpy
                        group.create_dataset(
                            key,
                            data=data[start_ptr:end_ptr, env_idx],
                            dtype=data.dtype,
                            compression="gzip",
                            compression_opts=5,
                        )
                    elif key == "seg":
                        group.create_dataset(
                            key,
                            data=data[start_ptr:end_ptr, env_idx],
                            dtype=data.dtype,
                            compression="gzip",
                            compression_opts=5,
                        )
                    else:
                        group.create_dataset(
                            key, data=data[start_ptr:end_ptr, env_idx], dtype=data.dtype
                        )

            # Observations need special processing
            if isinstance(self._trajectory_buffer.observation, dict):
                recursive_add_to_h5py(group, self._trajectory_buffer.observation, "obs")
            elif isinstance(self._trajectory_buffer.observation, np.ndarray):
                group.create_dataset(
                    "obs",
                    data=self._trajectory_buffer.observation[
                        start_ptr:end_ptr, env_idx
                    ],
                    dtype=self._trajectory_buffer.observation.dtype,
                )
            else:
                raise NotImplementedError(
                    f"RecordEpisode wrapper does not know how to handle observation data of type {type(self._trajectory_buffer.observation)}"
                )

            episode_info = dict(
                episode_id=self._episode_id,
                episode_seed=self._base_env._episode_seed,
                control_mode=self._base_env.control_mode,
                elapsed_steps=end_ptr - start_ptr - 1,
            )
            if self.num_envs == 1:
                # TODO (stao): handle reset kwargs/determinism for multiple envs somehow...
                episode_info.update(reset_kwargs=self.last_reset_kwargs)

            actions = self._trajectory_buffer.action[start_ptr + 1 : end_ptr, env_idx]
            terminated = self._trajectory_buffer.terminated[
                start_ptr + 1 : end_ptr, env_idx
            ]
            truncated = self._trajectory_buffer.truncated[
                start_ptr + 1 : end_ptr, env_idx
            ]
            group.create_dataset("actions", data=actions, dtype=np.float32)
            group.create_dataset("terminated", data=terminated, dtype=bool)
            group.create_dataset("truncated", data=truncated, dtype=bool)

            if self._trajectory_buffer.success is not None:
                group.create_dataset(
                    "success",
                    data=self._trajectory_buffer.success[
                        start_ptr + 1 : end_ptr, env_idx
                    ],
                    dtype=bool,
                )
                episode_info.update(
                    success=self._trajectory_buffer.success[end_ptr - 1, env_idx]
                )
            if self._trajectory_buffer.fail is not None:
                group.create_dataset(
                    "fail",
                    data=self._trajectory_buffer.fail[start_ptr + 1 : end_ptr, env_idx],
                    dtype=bool,
                )
                episode_info.update(
                    fail=self._trajectory_buffer.success[end_ptr - 1, env_idx]
                )
            recursive_add_to_h5py(group, self._trajectory_buffer.state, "env_states")

            if self.record_reward:
                group.create_dataset(
                    "rewards",
                    data=self._trajectory_buffer.reward[
                        start_ptr + 1 : end_ptr, env_idx
                    ],
                    dtype=np.float32,
                )

            self._json_data["episodes"].append(episode_info)
            dump_json(self._json_path, self._json_data, indent=2)
            flush_count += 1

        if verbose:
            if flush_count == 1:
                print(f"Recorded episode {self._episode_id}")
            else:
                print(
                    f"Recorded episodes {self._episode_id - flush_count} to {self._episode_id}"
                )
        # truncate self._trajectory_buffer down to save memory

        if flush_count > 0:
            self._trajectory_buffer.env_episode_ptr[env_idxs_to_flush] = (
                len(self._trajectory_buffer.done) - 1
            )
            min_env_ptr = self._trajectory_buffer.env_episode_ptr.min()
            N = len(self._trajectory_buffer.done)
            self._trajectory_buffer.state = slice_dict_array(
                self._trajectory_buffer.state, slice(min_env_ptr, N)
            )
            self._trajectory_buffer.observation = slice_dict_array(
                self._trajectory_buffer.observation, slice(min_env_ptr, N)
            )
            self._trajectory_buffer.action = slice_dict_array(
                self._trajectory_buffer.action, slice(min_env_ptr, N)
            )
            self._trajectory_buffer.reward = slice_dict_array(
                self._trajectory_buffer.reward, slice(min_env_ptr, N)
            )
            self._trajectory_buffer.terminated = slice_dict_array(
                self._trajectory_buffer.terminated, slice(min_env_ptr, N)
            )
            self._trajectory_buffer.truncated = slice_dict_array(
                self._trajectory_buffer.truncated, slice(min_env_ptr, N)
            )
            self._trajectory_buffer.done = slice_dict_array(
                self._trajectory_buffer.done, slice(min_env_ptr, N)
            )
            if self._trajectory_buffer.success is not None:
                self._trajectory_buffer.success = slice_dict_array(
                    self._trajectory_buffer.success, slice(min_env_ptr, N)
                )
            if self._trajectory_buffer.fail is not None:
                self._trajectory_buffer.fail = slice_dict_array(
                    self._trajectory_buffer.fail, slice(min_env_ptr, N)
                )
            self._trajectory_buffer.env_episode_ptr -= min_env_ptr

    def flush_video(self, suffix="", verbose=False, ignore_empty_transition=True):
        if len(self._render_images) == 0:
            return
        if ignore_empty_transition and len(self._render_images) == 1:
            return
        self._video_id += 1
        video_name = "{}".format(self._video_id)
        if suffix:
            video_name += "_" + suffix
        images_to_video(
            self._render_images,
            str(self.output_dir),
            video_name=video_name,
            fps=self.video_fps,
            verbose=verbose,
        )
        self._video_steps = 0
        self._render_images = []

    def close(self) -> None:
        if self._closed:
            # There is some strange bug when vector envs using record wrapper are closed/deleted, this code runs twice
            return
        self._closed = True
        if self.save_trajectory:
            # Handle the last episode only when `save_on_reset=True`
            if self.save_on_reset:
                self.flush_trajectory(
                    ignore_empty_transition=True,
                    env_idxs_to_flush=np.arange(self.num_envs),
                )
            if self.clean_on_close:
                clean_trajectories(self._h5_file, self._json_data)
                dump_json(self._json_path, self._json_data, indent=2)
            self._h5_file.close()
        if self.save_video:
            if self.save_on_reset:
                self.flush_video()
        return super().close()
