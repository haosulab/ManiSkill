"""Replay ManiSkill trajectories stored in HDF5 (.h5) format

The replayed trajectory can use different observation modes and control modes.

We support translating actions from certain controllers to a limited number of controllers.

The script is only tested for Panda, and may include some Panda-specific hardcode.
"""

import copy
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Annotated, Optional

import gymnasium as gym
import h5py
import numpy as np
import torch
import tyro
from tqdm import tqdm

import mani_skill.envs
from mani_skill.envs.utils.system.backend import CPU_SIM_BACKENDS
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.trajectory.merge_trajectory import merge_trajectories
from mani_skill.trajectory.utils.actions import conversion as action_conversion
from mani_skill.utils import common, io_utils, wrappers
from mani_skill.utils.logging_utils import logger
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
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
    Currently the Panda robots are the best supported when it comes to control mode conversion. Furthermore control mode conversion is not supported in GPU parallelized environments.
    """
    verbose: bool = False
    """Whether to print verbose information during trajectory replays"""
    save_traj: bool = False
    """Whether to save trajectories to disk. This will not override the original trajectory file."""
    save_video: bool = False
    """Whether to save videos"""
    max_retry: int = 0
    """Maximum number of times to try and replay a trajectory until the task reaches a success state at the end."""
    discard_timeout: bool = False
    """Whether to discard episodes that timeout and are truncated (depends on the max_episode_steps parameter of task)"""
    allow_failure: bool = False
    """Whether to include episodes that fail in saved videos and trajectory data based on the environment's evaluation returned "success" label"""
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
    shader: Optional[str] = None
    """Change shader used for rendering for all cameras. Default is none meaning it will use whatever was used in the original data collection or the environment default.
    Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer"""
    video_fps: Optional[int] = None
    """The FPS of saved videos. Defaults to the control frequency"""
    render_mode: str = "rgb_array"
    """The render mode used for saving videos. Typically there is also 'sensors' and 'all' render modes which further render all sensor outputs like cameras."""

    num_envs: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 1
    """Number of environments to run to replay trajectories. With CPU backends typically this is parallelized via python multiprocessing.
    For parallelized simulation backends like physx_gpu, this is parallelized within a single python process by leveraging the GPU."""


@dataclass
class ReplayResult:
    num_replays: int
    successful_replays: int


def sanity_check_and_format_seed(episode):
    """sanity checks the trajectory seed aligns with the episode seed. reformats the reset kwargs seed if missing or formatted wrong"""
    if "seed" in episode["reset_kwargs"]:
        if isinstance(episode["reset_kwargs"]["seed"], list):

            assert (
                len(episode["reset_kwargs"]["seed"]) == 1
            ), f"found multiple seeds for one trajectory (id={episode['episode_id']}) in the reset kwargs which means it is ambiguous which seed to use"
            episode["reset_kwargs"]["seed"] = episode["reset_kwargs"]["seed"][0]
        assert (
            episode["reset_kwargs"]["seed"] == episode["episode_seed"]
        ), f"found mismatch between trajectory seed and episode seed (id={episode['episode_id']})"
    else:
        episode["reset_kwargs"]["seed"] = episode["episode_seed"]


def replay_parallelized_sim(
    args: Args, env: RecordEpisode, pbar, episodes, trajectories
):
    pbar.reset(total=len(episodes))
    warned_reset_kwargs_options = False
    # split all episodes into batches of args.num_envs environments and process each batch in parallel, truncating where necessary
    # add fake episode padding to the end of the episodes to make sure all batches are the same size
    episode_pad = (args.num_envs - len(episodes) % args.num_envs) % args.num_envs
    batches = np.pad(
        np.array(episodes),
        (0, episode_pad),
        mode="constant",
        constant_values=episodes[-1],
    ).reshape(-1, args.num_envs)

    successful_replays = 0
    if pbar is not None:
        pbar.reset(total=len(episodes))
    for episode_batch_index, episode_batch in enumerate(batches):
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
        for i, trajectory_id in enumerate(trajectory_ids):

            # sanity check seeds and warn user if reset kwargs includes options (which are not supported in GPU sim replay)
            traj = trajectories[f"traj_{trajectory_id}"]
            episode = episode_batch[i]
            sanity_check_and_format_seed(episode)
            if not warned_reset_kwargs_options and "options" in episode["reset_kwargs"]:
                logger.warning(
                    f"Reset kwargs includes options, which are not supported in GPU sim replay and will be ignored."
                )
                warned_reset_kwargs_options = True

            # note (stao): this code to reformat the trajectories into a list of batched dicts can be optimized
            env_states = trajectory_utils.dict_to_list_of_dicts(traj["env_states"])
            actions = np.array(traj["actions"])

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
            # mark the fake padding trajectories as flushed
            if episode_batch_index == len(batches) - 1 and episode_pad > 0:
                flushed_trajectories[-episode_pad:] = True
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

                if args.save_traj:
                    envs_to_flush = (t >= episode_lens - 1) & (~flushed_trajectories)
                    flushed_trajectories |= envs_to_flush
                    if envs_to_flush.sum() > 0:
                        pbar.update(n=envs_to_flush.sum())
                        if not args.allow_failure:
                            if "success" in info:
                                envs_to_flush &= (info["success"] == True).cpu().numpy()
                        if args.discard_timeout:
                            envs_to_flush &= (truncated == False).cpu().numpy()
                        successful_replays += envs_to_flush.sum()
                        env.flush_trajectory(
                            env_idxs_to_flush=np.where(envs_to_flush)[0]
                        )
        else:
            raise NotImplementedError(
                "Replay with different control modes are not supported when replaying on GPU parallelized environments"
            )
    return ReplayResult(
        num_replays=len(episodes), successful_replays=successful_replays
    )


def replay_cpu_sim(
    args: Args, env: RecordEpisode, ori_env, pbar, episodes, trajectories
):
    successful_replays = 0
    for episode in episodes:
        sanity_check_and_format_seed(episode)
        episode_id = episode["episode_id"]
        traj_id = f"traj_{episode_id}"
        reset_kwargs = episode["reset_kwargs"]
        ori_control_mode = episode["control_mode"]
        if pbar is not None:
            pbar.set_description(f"Replaying {traj_id}")
        if traj_id not in trajectories:
            tqdm.write(f"{traj_id} does not exist in {args.traj_path}")
            continue

        for _ in range(args.max_retry + 1):
            # Each trial for each trajectory to replay, we reset the environment
            # and optionally set the first environment state
            env.reset(**reset_kwargs)
            if ori_env is not None:
                ori_env.reset(**reset_kwargs)

            # set first environment state and update recorded env state
            if args.use_first_env_state or args.use_env_states:
                ori_env_states = trajectory_utils.dict_to_list_of_dicts(
                    trajectories[traj_id]["env_states"]
                )
                if ori_env is not None:
                    ori_env.set_state_dict(ori_env_states[0])
                env.base_env.set_state_dict(ori_env_states[0])
                ori_env_states = ori_env_states[1:]
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
                        env._trajectory_buffer.state, common.batch(ori_env_states[0])
                    )
                    fixed_obs = env.base_env.get_obs()
                    recursive_replace(
                        env._trajectory_buffer.observation,
                        common.to_numpy(common.batch(fixed_obs)),
                    )
            # Original actions to replay
            ori_actions = trajectories[traj_id]["actions"][:]
            info = {}

            # Without conversion between control modes
            assert (
                args.target_control_mode is None
                or ori_control_mode == args.target_control_mode
                or not args.use_env_states
            ), "Cannot use env states when trying to \
                convert from one control mode to another. This is because control mode conversion causes there to be changes \
                in how many actions are taken to achieve the same states"
            if (
                args.target_control_mode is None
                or ori_control_mode == args.target_control_mode
            ):
                n = len(ori_actions)
                if pbar is not None:
                    pbar.reset(total=n)
                for t, a in enumerate(ori_actions):
                    if pbar is not None:
                        pbar.update()
                    _, _, _, truncated, info = env.step(a)
                    if args.use_env_states:
                        env.base_env.set_state_dict(ori_env_states[t])
                    if args.vis:
                        env.base_env.render_human()

            # From joint position to others
            elif ori_control_mode == "pd_joint_pos":
                info = action_conversion.from_pd_joint_pos(
                    args.target_control_mode,
                    ori_actions,
                    ori_env,
                    env,
                    render=args.vis,
                    pbar=pbar,
                    verbose=args.verbose,
                )

            # From joint delta position to others
            elif ori_control_mode == "pd_joint_delta_pos":
                info = action_conversion.from_pd_joint_delta_pos(
                    args.target_control_mode,
                    ori_actions,
                    ori_env,
                    env,
                    render=args.vis,
                    pbar=pbar,
                    verbose=args.verbose,
                )
            else:
                raise NotImplementedError(
                    f"Script currently does not support converting {ori_control_mode} to {args.target_control_mode}"
                )

            success = info.get("success", False)
            if args.discard_timeout:
                success = success and (not truncated)

            if success or args.allow_failure:
                successful_replays += 1
                if args.save_traj:
                    env.flush_trajectory()
                if args.save_video:
                    env.flush_video(ignore_empty_transition=False)
                break
            else:
                if args.verbose:
                    print("info", info)
        else:
            env.flush_video(save=False)
            tqdm.write(f"Episode {episode_id} is not replayed successfully. Skipping")

    return ReplayResult(
        num_replays=len(episodes), successful_replays=successful_replays
    )


def _main_helper(x):
    return _main(*x)


def _main(
    args: Args,
    use_cpu_backend,
    env_id,
    env_kwargs,
    ori_env_kwargs,
    record_episode_kwargs,
    proc_id: int = 0,
    num_procs=1,
):
    pbar = tqdm(position=proc_id, leave=None, unit="step", dynamic_ncols=True)

    # Load HDF5 containing trajectories
    traj_path = args.traj_path
    ori_h5_file = h5py.File(traj_path, "r")

    # Load associated json
    json_path = traj_path.replace(".h5", ".json")
    json_data = io_utils.load_json(json_path)
    env = gym.make(env_id, **env_kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        logger.warning(
            "We currently do not track which wrappers are used when recording trajectories but majority of the time in multi-agent envs with dictionary action spaces the actions are stored as flat vectors. We will flatten the action space with the ManiSkill provided FlattenActionSpaceWrapper. If you do not want this behavior you can copy the replay trajectory code yourself and modify it as needed."
        )
        env = FlattenActionSpaceWrapper(env)
    # TODO (support adding wrappers to the recorded data?)

    # if pbar is not None:
    #     pbar.set_postfix(
    #         {
    #             "control_mode": env_kwargs.get("control_mode"),
    #             "obs_mode": env_kwargs.get("obs_mode"),
    #         }
    #     )

    ### Prepare for recording ###

    # note for maniskill trajectory datasets the general naming format is <trajectory_name>.<obs_mode>.<control_mode>.<sim_backend>.h5
    # If it is called <file_name>.h5 then we assume obs_mode=None, control_mode=pd_joint_pos, and sim_backend=physx_cpu
    output_dir = os.path.dirname(traj_path)
    ori_traj_name = os.path.splitext(os.path.basename(traj_path))[0]
    parts = ori_traj_name.split(".")
    if len(parts) > 1:
        ori_traj_name = parts[0]
    suffix = "{}.{}.{}".format(
        env.unwrapped.obs_mode,
        env.unwrapped.control_mode,
        env.unwrapped.backend.sim_backend,
    )
    new_traj_name = ori_traj_name + "." + suffix
    if use_cpu_backend:
        if num_procs > 1:
            new_traj_name = new_traj_name + "." + str(proc_id)
        if args.target_control_mode is not None:
            ori_env = gym.make(env_id, **ori_env_kwargs)
        else:
            ori_env = None
    else:
        pass

    env = wrappers.RecordEpisode(
        env,
        output_dir,
        trajectory_name=new_traj_name,
        video_fps=(
            args.video_fps if args.video_fps is not None else env.unwrapped.control_freq
        ),
        **record_episode_kwargs,
    )

    if env.save_trajectory:
        output_h5_path = env._h5_file.filename
        assert not os.path.samefile(output_h5_path, traj_path)
    else:
        output_h5_path = None

    episodes = json_data["episodes"][: args.count]
    if use_cpu_backend:
        inds = np.arange(len(episodes))
        inds = np.array_split(inds, num_procs)[proc_id]
        replay_result = replay_cpu_sim(
            args, env, ori_env, pbar, [episodes[index] for index in inds], ori_h5_file
        )
    else:
        replay_result = replay_parallelized_sim(args, env, pbar, episodes, ori_h5_file)

    env.close()
    ori_h5_file.close()
    return output_h5_path, replay_result


def parse_args(args=None):
    return tyro.cli(Args, args=args)


def main(args: Args):
    traj_path = args.traj_path
    # Load trajectory metadata json
    json_path = traj_path.replace(".h5", ".json")
    json_data = io_utils.load_json(json_path)

    env_info = json_data["env_info"]
    env_id = env_info["env_id"]
    ori_env_kwargs = env_info["env_kwargs"]
    env_kwargs = ori_env_kwargs.copy()

    ### Checks and setting up env kwargs ###
    # First we determine how to setup the environment to replay demonstrations and raise relevant warnings to the user
    if (
        "sim_backend" in ori_env_kwargs
        and ori_env_kwargs["sim_backend"] != args.sim_backend
        and args.use_env_states
    ):
        logger.warning(
            f"Warning: Using different backend ({args.sim_backend}) than the original used to collect the trajectory data "
            f"({ori_env_kwargs['sim_backend']}). This may cause replay failures due to "
            f"differences in simulation/physics engine backend. Use the same backend by passing -b {ori_env_kwargs['sim_backend']} "
            f"or replay by environment states by passing --use-env-states instead."
        )
    if args.sim_backend is None:
        # try to guess which sim backend to use
        if "sim_backend" not in ori_env_kwargs:
            args.sim_backend = "physx_cpu"
        else:
            args.sim_backend = ori_env_kwargs["sim_backend"]

    ori_env_kwargs["sim_backend"] = args.sim_backend
    env_kwargs["sim_backend"] = args.sim_backend

    # modify the env kwargs according to the users inputs
    target_obs_mode = args.obs_mode
    target_control_mode = args.target_control_mode
    if target_obs_mode is not None:
        env_kwargs["obs_mode"] = target_obs_mode
    if target_control_mode is not None:
        env_kwargs["control_mode"] = target_control_mode
    if args.shader is not None:
        env_kwargs["shader_dir"] = args.shader  # change all shaders
    env_kwargs["reward_mode"] = args.reward_mode
    env_kwargs[
        "render_mode"
    ] = (
        args.render_mode
    )  # note this only affects the videos saved as RecordEpisode wrapper calls env.render

    record_episode_kwargs = dict(
        save_on_reset=False,
        save_trajectory=args.save_traj,
        save_video=args.save_video,
        record_reward=args.record_rewards,
    )

    if args.count is not None and args.count > len(json_data["episodes"]):
        logger.warning(
            f"Warning: Requested to replay {args.count} demos but there are only {len(json_data['episodes'])} demos collected, replaying all demos now"
        )
        args.count = len(json_data["episodes"])
    elif args.count is None:
        args.count = len(json_data["episodes"])

    pbar = tqdm(total=args.count, unit="step", dynamic_ncols=True)

    # if missing info or auto sim backend is provided, we try to infer which backend is being used
    if "sim_backend" not in env_kwargs or (
        env_kwargs["sim_backend"] == "auto"
        and ("num_envs" not in env_kwargs or env_kwargs["num_envs"] == 1)
    ):
        env_kwargs["sim_backend"] = "physx_cpu"
    env_kwargs["num_envs"] = args.num_envs
    if env_kwargs["sim_backend"] not in CPU_SIM_BACKENDS:
        record_episode_kwargs["max_steps_per_video"] = env_info["max_episode_steps"]
        _, replay_result = _main(
            args,
            use_cpu_backend=False,
            env_id=env_id,
            env_kwargs=env_kwargs,
            ori_env_kwargs=ori_env_kwargs,
            record_episode_kwargs=record_episode_kwargs,
            proc_id=0,
            num_procs=1,
        )

    else:
        env_kwargs["num_envs"] = 1
        ori_env_kwargs["num_envs"] = 1
        if args.num_envs > 1:
            pool = mp.Pool(args.num_envs)
            proc_args = [
                (
                    copy.deepcopy(args),
                    True,
                    env_id,
                    env_kwargs,
                    ori_env_kwargs,
                    record_episode_kwargs,
                    i,
                    args.num_envs,
                )
                for i in range(args.num_envs)
            ]
            # res = pool.starmap(_main, proc_args)
            res = list(tqdm(pool.imap(_main_helper, proc_args), total=args.count))
            replay_results_list = [x[1] for x in res]
            trajectory_paths = [x[0] for x in res]
            pool.close()
            if args.save_traj:
                # A hack to find the path
                output_path = trajectory_paths[0][: -len("0.h5")] + "h5"
                merge_trajectories(output_path, trajectory_paths)
                for h5_path in trajectory_paths:
                    tqdm.write(f"Remove {h5_path}")
                    os.remove(h5_path)
                    json_path = h5_path.replace(".h5", ".json")
                    tqdm.write(f"Remove {json_path}")
                    os.remove(json_path)
            replay_result = ReplayResult(
                num_replays=sum([x.num_replays for x in replay_results_list]),
                successful_replays=sum(
                    [x.successful_replays for x in replay_results_list]
                ),
            )
        else:
            _, replay_result = _main(
                args,
                use_cpu_backend=True,
                env_id=env_id,
                env_kwargs=env_kwargs,
                ori_env_kwargs=ori_env_kwargs,
                record_episode_kwargs=record_episode_kwargs,
                proc_id=0,
                num_procs=1,
            )

    pbar.close()
    print(
        f"Replayed {replay_result.num_replays} episodes, "
        f"{replay_result.successful_replays}/{replay_result.num_replays}={replay_result.successful_replays/replay_result.num_replays*100:.2f}% demos saved"
    )


if __name__ == "__main__":
    # spawn is needed due to warp init issue
    mp.set_start_method("spawn")
    main(parse_args())
