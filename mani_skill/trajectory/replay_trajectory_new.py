"""Replay ManiSkill trajectories stored in HDF5 (.h5) format

The replayed trajectory can use different observation modes and control modes.

We support translating actions from certain controllers to a limited number of controllers.

The script is only tested for Panda, and may include some Panda-specific hardcode.
"""

import os
from dataclasses import dataclass
from typing import Annotated, Optional

import gymnasium as gym
import h5py
import numpy as np
import torch
import tyro
from tqdm.auto import tqdm

import mani_skill.envs
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.trajectory.merge_trajectory import merge_trajectories
from mani_skill.trajectory.utils.actions import conversion as action_conversion
from mani_skill.utils import common, io_utils, wrappers
from mani_skill.utils.logging_utils import logger
from mani_skill.utils.wrappers.record import RecordEpisode


@dataclass
class Args:
    traj_path: str
    """Path to the trajectory .h5 file to replay"""
    sim_backend: Annotated[Optional[str], tyro.conf.arg(aliases=["-b"])] = None
    """Which simulation backend to use. Can be 'physx_cpu', 'physx_gpu'. If not specified the backend used is the same as the one used to collect the trajectory data."""
    obs_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-o"])] = None
    """Target observation mode to record in the trajectory. See
    https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html for a full list of supported observation modes."""
    target_control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = None
    """Target control mode to convert the demonstration actions to.
    Note that not all control modes can be converted to others successfully and not all robots have easy to convert control modes.
    Currently the Panda robots are the best supported when it comes to control mode conversion.
    """
    verbose: bool = False
    """Whether to print verbose information during trajectory replays"""
    save_traj: bool = False
    """Whether to save trajectories to disk. This will not override the original trajectory file."""
    save_video: bool = False
    """Whether to save videos"""
    num_procs: int = 1
    """Number of processes to use to help parallelize the trajectory replay process. This argument is the same as num_envs for the CPU backend and is kept for backwards compatibility."""
    max_retry: int = 0
    """Maximum number of times to try and replay a trajectory until the task reaches a success state at the end."""
    discard_timeout: bool = False
    """Whether to discard episodes that timeout and are truncated (depends on the max_episode_steps parameter of task)"""
    allow_failure: bool = False
    """Whether to include episodes that fail in saved videos and trajectory data"""
    vis: bool = False
    """Whether to visualize the trajectory replay via the GUI."""
    use_env_states: bool = False
    """Whether to replay by environment states instead of actions. This guarantees that the environment will look exactly
    the same as the original trajectory at every step."""
    use_first_env_state: bool = False
    """Use the first env state in the trajectory to set initial state. This can be useful for trying to replay
    demonstrations collected in the CPU simulation in the GPU simulation by first starting with the same initial
    state as GPU simulated tasks will randomize initial states differently despite given the same seed compared to CPU sim."""
    count: Optional[int] = None
    """Number of demonstrations to replay before exiting. By default will replay all demonstrations"""
    reward_mode: Optional[str] = None
    """Specifies the reward type that the env should use. By default it will pick the first supported reward mode. Most environments
    support 'sparse', 'none', and some further support 'normalized_dense' and 'dense' reward modes"""
    record_rewards: bool = False
    """Whether the replayed trajectory should include rewards"""
    shader: str = "default"
    """Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing
    and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""
    video_fps: Optional[int] = None
    """The FPS of saved videos. Defaults to the control frequency"""
    render_mode: str = "rgb_array"
    """The render mode used for saving videos. Typically there is also 'sensors' and 'all' render modes which further render all sensor outputs like cameras."""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run to replay trajectories. With CPU backends typically this is parallelized via python multiprocessing.
    For parallelized simulation backends like physx_gpu, this is parallelized within a single python process by leveraging the GPU."""


def replay_parallelized_sim(
    args: Args, env: RecordEpisode, pbar, episodes, trajectories
):
    # split all episodes into batches of args.num_envs environments and process each batch in parallel, truncating where necessary
    # add fake episode padding to the end of the episodes to make sure all batches are the same size
    episodes = np.array(episodes)
    # episodes = np.pad(episodes, (0, args.num_envs - len(episodes) % args.num_envs), mode='constant', constant_values=None)

    # Pad array to be divisible by num_envs and reshape into batches
    n_pad = (args.num_envs - len(episodes) % args.num_envs) % args.num_envs
    batches = np.pad(
        episodes, (0, n_pad), mode="constant", constant_values=episodes[-1]
    ).reshape(-1, args.num_envs)
    if pbar is not None:
        pbar.reset(total=len(episodes))
    for episode_batch in batches:
        import ipdb

        ipdb.set_trace()
        trajectory_ids = [episode["episode_id"] for episode in episode_batch]
        episode_lens = np.array([episode["elapsed_steps"] for episode in episode_batch])
        ori_control_mode = episode_batch[0]["control_mode"]
        assert all(
            [episode["control_mode"] == ori_control_mode for episode in episode_batch]
        ), "Replay trajectory with parallelized environments is only supported for trajectories with the same control mode"
        episode_batch_max_len = max(episode_lens)
        seeds = torch.tensor(
            [episode["episode_seed"] for episode in episode_batch],
            device=env.base_env.device,
        )
        env.reset(seed=seeds)

        # generate batched env states and actions
        env_states_list = []
        original_actions_batch = []
        env_states_batch = []  # list of batched env states shape (max_steps, D)
        for trajectory_id in trajectory_ids:
            env_states = trajectory_utils.dict_to_list_of_dicts(
                trajectories[f"traj_{trajectory_id}"]["env_states"]
            )
            actions = np.array(trajectories[f"traj_{trajectory_id}"]["actions"])

            # padding
            for _ in range(episode_batch_max_len + 1 - len(env_states)):
                env_states.append(env_states[-1])
            if len(actions) < episode_batch_max_len:
                actions = np.concatenate(
                    [
                        actions,
                        np.zeros(
                            (episode_batch_max_len - len(actions), actions.shape[1])
                        ),
                    ],
                    axis=0,
                )
            env_states_list.append(env_states)
            original_actions_batch.append(actions)
        for t in range(episode_batch_max_len + 1):
            env_states_batch.append(
                trajectory_utils.list_of_dicts_to_dict(
                    [env_states_list[i][t] for i in range(len(env_states_list))]
                )
            )

        original_actions_batch = np.stack(original_actions_batch, axis=1)
        if args.use_first_env_state or args.use_env_states:
            # set the first environment state to the first states in the trajectories given
            env.base_env.set_state_dict(env_states_batch[0])
            if args.save_traj:
                # replace the first saved env state
                # since we set state earlier and RecordEpisode will save the reset to state.
                def recursive_replace(x, y):
                    if isinstance(x, np.ndarray):
                        x[-1, :] = y[-1, :]
                    else:
                        for k in x.keys():
                            recursive_replace(x[k], y[k])

                recursive_replace(
                    env._trajectory_buffer.state, common.batch(env_states_batch[0])
                )
                recursive_replace(
                    env._trajectory_buffer.observation,
                    common.to_numpy(common.batch(env.base_env.get_obs())),
                )

        # replay with env states / actions
        if (
            args.target_control_mode is None
            or ori_control_mode == args.target_control_mode
        ):
            flushed_trajectories = np.zeros(len(episode_batch), dtype=bool)
            for t, a in enumerate(original_actions_batch):
                _, _, _, truncated, info = env.step(a)
                if args.use_env_states:
                    # NOTE (stao): due to the high precision nature of some tasks even taking a single step in GPU simulation (in e.g. PushT-v1) can lead
                    # to some non-deterministic behaviors leading to some steps labeled with slightly wrong observations/rewards/success/fail data (1e-4 error).
                    # I unfortunately do not have a good solution for this apart from using the same number of parallel environments to replay demos as the original trajectory collection.
                    env.base_env.set_state_dict(env_states_batch[t])
                if args.vis:
                    env.base_env.render_human()
                # if the elapsed_steps mark saved in the trajectory is reached for any env, flush that trajectory buffer

                envs_to_flush = (t >= episode_lens - 1) & (~flushed_trajectories)
                if envs_to_flush.sum() > 0:
                    pbar.update(n=envs_to_flush.sum())
                    env.flush_trajectory(env_idxs_to_flush=np.where(envs_to_flush)[0])
                flushed_trajectories |= envs_to_flush


def parse_args(args=None):
    return tyro.cli(Args, args=args)


def _main(args: Args, proc_id: int = 0, num_procs=1, pbar=None):
    pbar = tqdm(position=proc_id, leave=None, unit="step", dynamic_ncols=True)

    # Load HDF5 containing trajectories
    traj_path = args.traj_path
    ori_h5_file = h5py.File(traj_path, "r")

    # Load associated json
    json_path = traj_path.replace(".h5", ".json")
    json_data = io_utils.load_json(json_path)

    env_info = json_data["env_info"]
    env_id = env_info["env_id"]
    ori_env_kwargs = env_info["env_kwargs"]
    env_kwargs = ori_env_kwargs.copy()

    ### Environment Creation ###
    # First we determine how to setup the environment to replay demonstrations and raise relevant warnings to the user
    if ori_env_kwargs["sim_backend"] != args.sim_backend and args.use_env_states:
        logger.warning(
            f"Warning: Using different backend ({args.sim_backend}) than the original used to collect the trajectory data "
            f"({ori_env_kwargs['sim_backend']}). This may cause replay failures due to "
            f"differences in simulation/physics engine backend. Use the same backend by passing -b {ori_env_kwargs['sim_backend']} "
            f"or replay by environment states by passing --use-env-states instead."
        )
        ori_env_kwargs["sim_backend"] = args.sim_backend
        env_kwargs["sim_backend"] = args.sim_backend

    # modify the env kwargs according to the users inputs
    target_obs_mode = args.obs_mode
    target_control_mode = args.target_control_mode
    if target_obs_mode is not None:
        env_kwargs["obs_mode"] = target_obs_mode
    if target_control_mode is not None:
        env_kwargs["control_mode"] = target_control_mode
    env_kwargs["shader_dir"] = args.shader  # change all shaders
    env_kwargs["reward_mode"] = args.reward_mode
    env_kwargs[
        "render_mode"
    ] = (
        args.render_mode
    )  # note this only affects the videos saved as RecordEpisode wrapper calls env.render
    env_kwargs["num_envs"] = args.num_envs

    # create the original environment for replay
    # ori_env = gym.make(env_id, **ori_env_kwargs)
    env = gym.make(env_id, **env_kwargs)
    # TODO (support adding wrappers to the recorded data?)

    if pbar is not None:
        pbar.set_postfix(
            {
                "control_mode": env_kwargs.get("control_mode"),
                "obs_mode": env_kwargs.get("obs_mode"),
            }
        )

    ### Prepare for recording ###

    # note for maniskill trajectory datasets the general naming format is <trajectory_name>.<obs_mode>.<control_mode>.<sim_backend>.h5
    # If it is called <file_name>.h5 then we assume obs_mode=None, control_mode=pd_joint_pos, and sim_backend=physx_cpu

    output_dir = os.path.dirname(traj_path)
    ori_traj_name = os.path.splitext(os.path.basename(traj_path))[0]
    parts = ori_traj_name.split(".")
    if len(parts) > 1:
        ori_traj_name = parts[0]
    suffix = "{}.{}.{}".format(
        env.unwrapped.obs_mode, env.unwrapped.control_mode, env.unwrapped.device.type
    )
    new_traj_name = ori_traj_name + "." + suffix
    if num_procs > 1:
        new_traj_name = new_traj_name + "." + str(proc_id)
    env = wrappers.RecordEpisode(
        env,
        output_dir,
        save_on_reset=False,
        save_trajectory=args.save_traj,
        trajectory_name=new_traj_name,
        save_video=args.save_video,
        video_fps=args.video_fps
        if args.video_fps is not None
        else env.unwrapped.control_freq,
        record_reward=args.record_rewards,
        max_steps_per_video=env_info["max_episode_steps"]
        if args.num_envs > 1
        else None,
    )

    if env.save_trajectory:
        output_h5_path = env._h5_file.filename
        assert not os.path.samefile(output_h5_path, traj_path)
    else:
        output_h5_path = None

    episodes = json_data["episodes"][: args.count]
    replay_parallelized_sim(args, env, pbar, episodes, ori_h5_file)
    # n_ep = len(episodes)
    # inds = np.arange(n_ep)
    # inds = np.array_split(inds, num_procs)[proc_id]
    # import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    args = parse_args()
    _main(args)
