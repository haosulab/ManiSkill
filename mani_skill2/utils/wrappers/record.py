import copy
import time
from pathlib import Path
from typing import List, Optional, Union

import gymnasium as gym
import h5py
import numpy as np
import sapien.physx as physx
from gymnasium import spaces

from mani_skill2 import get_commit_info, logger
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.sapien_utils import to_numpy

from ..common import extract_scalars_from_info, flatten_dict_keys
from ..io_utils import dump_json
from ..visualization.misc import images_to_video, put_info_on_image, tile_images


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
        max_episode_steps=env.spec.max_episode_steps,
        env_kwargs=env_kwargs,
    )


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


def pack_step_data(state, obs, action, rew, terminated, truncated, info):
    data = dict(
        s=to_numpy(state) if state is not None else None,
        o=copy.deepcopy(to_numpy(obs)) if obs is not None else None,
        a=to_numpy(action) if action is not None else None,
        r=to_numpy(rew) if rew is not None else None,
        terminated=to_numpy(terminated) if terminated is not None else None,
        truncated=to_numpy(truncated) if truncated is not None else None,
        info=to_numpy(info),
    )
    return data


class RecordEpisode(gym.Wrapper):
    """Record trajectories or videos for episodes.

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
    - env_init_state: [D], `np.float32`. The initial environment state. It is used for soft-body environments, since their states (particle positions) can use too much space.
    - obs (optional): observations. If the observation is a `dict`, the value will be stored in `obs/{key}`. The convention is applied recursively for nested dict.


    Args:
        env: gym.Env
        output_dir: output directory
        save_trajectory: whether to save trajectory
        trajectory_name: name of trajectory file (.h5). Use timestamp if not provided.
        save_video: whether to save video
        render_mode: rendering mode passed to `env.render`
        save_on_reset: whether to save the previous trajectory automatically when resetting.
            If True, the trajectory with empty transition will be ignored automatically.
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
        clean_on_close=True,
        record_reward=False,
        init_state_only=False,
        video_fps=20,
    ):
        # NOTE (stao): don't worry about replay by action, not needed really, only replay by state for visual, otherwise just train directly.
        super().__init__(env)

        self.output_dir = Path(output_dir)
        self.init_state_only = init_state_only
        if save_trajectory or save_video:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_on_reset = save_on_reset
        self.video_fps = video_fps
        self._episode_id = -1
        self._episode_data = []
        self._episode_info = {}

        self.save_trajectory = save_trajectory
        if self._base_env.num_envs > 1:
            # TODO (stao): fix trajectory saving on gpu simulation.
            assert self.save_trajectory == False
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

        self.save_video = save_video
        self.info_on_video = info_on_video
        self._render_images = []
        if info_on_video and physx.is_gpu_enabled():
            raise ValueError(
                "Cannot turn info_on_video=True when using GPU simulation as the text would be too small"
            )
        self.video_nrows = int(np.sqrt(self.unwrapped.num_envs))

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
        options: Optional[dict] = None,
        **kwargs,
    ):
        skip_trajectory = False
        if options is not None:
            options.pop("save_trajectory", False)

        if self.save_on_reset and self._episode_id >= 0 and not skip_trajectory:
            self.flush_trajectory(ignore_empty_transition=True)
            # when to flush video? Use last parallel env done?
            self.flush_video(ignore_empty_transition=True)

        # Clear cache
        self._episode_data = []
        self._episode_info = {}
        self._render_images = []
        if not skip_trajectory:
            self._episode_id += 1

        reset_kwargs = copy.deepcopy(dict(seed=seed, options=options, **kwargs))
        obs, info = super().reset(*args, seed=seed, options=options, **kwargs)

        if self.save_trajectory:
            state = self._base_env.get_state()
            data = pack_step_data(state, obs, None, None, None, None, None)
            self._episode_data.append(data)
            self._episode_info.update(
                episode_id=self._episode_id,
                episode_seed=getattr(self.unwrapped, "_episode_seed", None),
                reset_kwargs=reset_kwargs,
                control_mode=getattr(self.unwrapped, "control_mode", None),
                elapsed_steps=0,
            )

        if self.save_video:
            self._render_images.append(self.capture_image())

        return obs, info

    def step(self, action):
        obs, rew, terminated, truncated, info = super().step(action)

        if self.save_trajectory:
            state = self.env.unwrapped.get_state()
            data = pack_step_data(state, obs, action, rew, terminated, truncated, info)
            self._episode_data.append(data)
            self._episode_info["elapsed_steps"] += 1
            self._episode_info["info"] = to_numpy(info)

        if self.save_video:
            image = self.capture_image()

            if self.info_on_video:
                scalar_info = extract_scalars_from_info(info)
                extra_texts = [
                    f"reward: {rew:.3f}",
                    "action: {}".format(",".join([f"{x:.2f}" for x in action])),
                ]
                image = put_info_on_image(image, scalar_info, extras=extra_texts)

            self._render_images.append(image)

        return obs, rew, terminated, truncated, info

    def flush_trajectory(self, verbose=False, ignore_empty_transition=False):
        if (
            not self.save_trajectory or len(self._episode_data) == 0
        ):  # TODO (stao): remove this, this is not intuitive as it depends on data in self.
            return
        if ignore_empty_transition and len(self._episode_data) == 1:
            return

        # find which trajectories completed
        traj_id = "traj_{}".format(self._episode_id)
        group = self._h5_file.create_group(traj_id, track_order=True)

        # Observations need special processing
        obs = [x["o"] for x in self._episode_data]
        if isinstance(obs[0], dict):
            if len(obs[0]) > 0:
                obs_group = group.create_group("obs", track_order=True)
                # NOTE(jigu): If each obs is empty, then nothing will be stored.
                obs = [flatten_dict_keys(x) for x in obs]
                obs = {k: [x[k] for x in obs] for k in obs[0].keys()}
                obs = {k: np.stack(v) for k, v in obs.items()}
                for k, v in obs.items():
                    # create subgroups if they don't exist yet. Can be removed once https://github.com/h5py/h5py/issues/1471 is fixed
                    subgroups = k.split("/")[:-1]
                    curr_group = obs_group
                    for subgroup in subgroups:
                        if subgroup in curr_group:
                            curr_group = curr_group[subgroup]
                        else:
                            curr_group = curr_group.create_group(
                                subgroup, track_order=True
                            )

                    if "rgb" in k and v.ndim == 4:
                        # NOTE(jigu): It is more efficient to use gzip than png for a sequence of images.
                        group.create_dataset(
                            "obs/" + k,
                            data=v,
                            dtype=v.dtype,
                            compression="gzip",
                            compression_opts=5,
                        )
                    elif "depth" in k and v.ndim in (3, 4):
                        # NOTE (stao): By default now cameras in ManiSkill return depth values of type uint16 for numpy
                        group.create_dataset(
                            "obs/" + k,
                            data=v,
                            dtype=v.dtype,
                            compression="gzip",
                            compression_opts=5,
                        )
                    elif "seg" in k and v.ndim in (3, 4):
                        assert (
                            np.issubdtype(v.dtype, np.integer) or v.dtype == np.bool_
                        ), v.dtype
                        group.create_dataset(
                            "obs/" + k,
                            data=v,
                            dtype=v.dtype,
                            compression="gzip",
                            compression_opts=5,
                        )
                    else:
                        group.create_dataset("obs/" + k, data=v, dtype=v.dtype)
        elif isinstance(obs[0], np.ndarray):
            obs = np.stack(obs)
            group.create_dataset("obs", data=obs, dtype=obs.dtype)
        else:
            print(obs[0])
            raise NotImplementedError(type(obs[0]))

        if len(self._episode_data) == 1:
            action_space = self.env.action_space
            assert isinstance(action_space, spaces.Box), action_space
            actions = np.empty(
                shape=(0,) + action_space.shape,
                dtype=action_space.dtype,
            )
            terminated = np.empty(shape=(0,), dtype=bool)
            truncated = np.empty(shape=(0,), dtype=bool)
        else:
            actions = np.stack([x["a"] for x in self._episode_data[1:]])
            terminated = np.stack([x["terminated"] for x in self._episode_data[1:]])
            truncated = np.stack([x["truncated"] for x in self._episode_data[1:]])
            if "success" in self._episode_data[1]["info"]:
                success = np.stack(
                    [x["info"]["success"] for x in self._episode_data[1:]]
                )
                group.create_dataset("success", data=success, dtype=bool)
            if "fail" in self._episode_data[1]["info"]:
                fail = np.stack([x["info"]["fail"] for x in self._episode_data[1:]])
                group.create_dataset("fail", data=fail, dtype=bool)

        # TODO (stao): Only support array like states at the moment
        env_states = np.stack([x["s"] for x in self._episode_data])

        # Dump
        group.create_dataset("actions", data=actions, dtype=np.float32)
        group.create_dataset("terminated", data=terminated, dtype=bool)
        group.create_dataset("truncated", data=truncated, dtype=bool)

        if self.record_reward:
            rewards = np.stack([x["r"] for x in self._episode_data]).astype(np.float32)
            group.create_dataset("rewards", data=rewards, dtype=np.float32)

        if self.init_state_only:
            group.create_dataset("env_init_state", data=env_states[0], dtype=np.float32)
        else:
            group.create_dataset("env_states", data=env_states, dtype=np.float32)

        # Handle JSON
        self._json_data["episodes"].append(self._episode_info)
        dump_json(self._json_path, self._json_data, indent=2)

        if verbose:
            print("Record the {}-th episode".format(self._episode_id))

    def flush_video(self, suffix="", verbose=False, ignore_empty_transition=False):
        if not self.save_video or len(self._render_images) == 0:
            return
        if ignore_empty_transition and len(self._render_images) == 1:
            return

        video_name = "{}".format(self._episode_id)
        if suffix:
            video_name += "_" + suffix
        images_to_video(
            self._render_images,
            str(self.output_dir),
            video_name=video_name,
            fps=self.video_fps,
            verbose=verbose,
        )

    def close(self) -> None:
        if self.save_trajectory:
            # Handle the last episode only when `save_on_reset=True`
            if self.save_on_reset:
                traj_id = "traj_{}".format(self._episode_id)
                if traj_id in self._h5_file:
                    logger.warning(f"{traj_id} exists in h5.")
                else:
                    self.flush_trajectory(ignore_empty_transition=True)
            if self.clean_on_close:
                clean_trajectories(self._h5_file, self._json_data)
                dump_json(self._json_path, self._json_data, indent=2)
            self._h5_file.close()
        if self.save_video:
            if self.save_on_reset:
                self.flush_video(ignore_empty_transition=True)
        return super().close()
