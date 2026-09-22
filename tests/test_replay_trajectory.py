import os

import gymnasium as gym
import h5py
import numpy as np
import pytest
import torch
from tqdm import tqdm

from mani_skill import ASSET_DIR
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.wrappers.record import RecordEpisode


@pytest.mark.parametrize(
    "control_mode",
    [
        "pd_joint_delta_pos",
        "pd_joint_target_delta_pos",
        "pd_joint_vel",
        "pd_ee_delta_pose",
    ],
)
def test_replay_trajectory(control_mode):
    env_id = "PickCube-v1"
    # from mani_skill.utils.download_demo import main as download_demo, parse_args as download_demo_parse_args
    # download_demo(download_demo_parse_args(args=[env_id]))
    from mani_skill.trajectory.replay_trajectory import main, parse_args

    main(
        parse_args(
            args=[
                "--traj-path",
                f"{os.path.expandvars('$HOME')}/.maniskill/demos/{env_id}/teleop/trajectory.h5",
                "--save-traj",
                "--target-control-mode",
                control_mode,
                "--count",
                "4",
            ]
        )
    )


def _collect_dummy_dataset(tmp_path, sim_backend, num_episodes=2, episode_len=5):
    """Collect a small dataset of (unsuccessful) PickCube episodes to use as replay input"""
    env = gym.make(
        "PickCube-v1",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        sim_backend=sim_backend,
    )
    env = RecordEpisode(
        env, output_dir=str(tmp_path), trajectory_name="trajectory", save_video=False
    )
    action = np.zeros(env.unwrapped.single_action_space.shape, dtype=np.float32)
    for seed in range(num_episodes):
        env.reset(seed=seed)
        for _ in range(episode_len):
            env.step(action)
    env.close()
    return str(tmp_path / "trajectory.h5")


def _patch_success_schedule(env, succeed_from_reset):
    """Force replays to report failure until the env has been reset `succeed_from_reset` times"""
    original_reset = env.reset
    original_step = env.step
    num_resets = 0

    def reset_and_count(*args, **kwargs):
        nonlocal num_resets
        num_resets += 1
        return original_reset(*args, **kwargs)

    def step_with_fake_success(action):
        obs, rew, terminated, truncated, info = original_step(action)
        info["success"] = torch.tensor(
            [num_resets >= succeed_from_reset], device=env.base_env.device
        )
        return obs, rew, terminated, truncated, info

    env.reset = reset_and_count
    env.step = step_with_fake_success


def _assert_single_clean_replay(tmp_path, source_episodes, episode_len):
    """The replay output must contain exactly one trajectory holding only the successful episode's data"""
    with h5py.File(tmp_path / "replay.h5", "r") as f:
        assert len(f.keys()) == 1
        assert len(f["traj_0"]["actions"]) == episode_len
    replay_episodes = load_json(tmp_path / "replay.json")["episodes"]
    assert len(replay_episodes) == 1
    assert replay_episodes[0]["elapsed_steps"] == episode_len
    assert replay_episodes[0]["episode_seed"] == source_episodes[1]["episode_seed"]


def test_replay_trajectory_failed_episode_not_saved_cpu(tmp_path):
    """Regression test: a failed replay attempt (allow_failure=False) must be discarded from the
    recording buffer instead of being concatenated into the next saved trajectory"""
    from mani_skill.trajectory.replay_trajectory import Args, replay_cpu_sim

    episode_len = 5
    traj_path = _collect_dummy_dataset(tmp_path, "physx_cpu", episode_len=episode_len)
    source_episodes = load_json(traj_path.replace(".h5", ".json"))["episodes"]

    env = gym.make(
        "PickCube-v1",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        sim_backend="physx_cpu",
    )
    env = RecordEpisode(
        env,
        output_dir=str(tmp_path),
        trajectory_name="replay",
        save_on_reset=False,
        save_video=False,
    )
    # first episode replay fails, second episode replay succeeds
    _patch_success_schedule(env, succeed_from_reset=2)

    args = Args(traj_path=traj_path, save_traj=True)
    with h5py.File(traj_path, "r") as trajectories:
        result = replay_cpu_sim(args, env, None, None, source_episodes, trajectories)
    env.close()

    assert result.successful_replays == 1
    _assert_single_clean_replay(tmp_path, source_episodes, episode_len)


@pytest.mark.gpu_sim
def test_replay_trajectory_failed_episode_not_saved_gpu(tmp_path):
    """Same regression test as above for the parallelized GPU simulation replay path"""
    from mani_skill.trajectory.replay_trajectory import Args, replay_parallelized_sim

    episode_len = 5
    traj_path = _collect_dummy_dataset(tmp_path, "physx_cuda", episode_len=episode_len)
    source_episodes = load_json(traj_path.replace(".h5", ".json"))["episodes"]

    env = gym.make(
        "PickCube-v1",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        sim_backend="physx_cuda",
    )
    env = RecordEpisode(
        env,
        output_dir=str(tmp_path),
        trajectory_name="replay",
        save_on_reset=False,
        save_video=False,
    )
    # first episode replay fails, second episode replay succeeds
    _patch_success_schedule(env, succeed_from_reset=2)

    args = Args(traj_path=traj_path, save_traj=True, num_envs=1)
    with h5py.File(traj_path, "r") as trajectories:
        result = replay_parallelized_sim(
            args, env, tqdm(disable=True), source_episodes, trajectories
        )
    env.close()

    assert result.successful_replays == 1
    _assert_single_clean_replay(tmp_path, source_episodes, episode_len)
