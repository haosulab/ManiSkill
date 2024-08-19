import gymnasium as gym
import pytest

import mani_skill.envs
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.utils.wrappers.flatten import (
    FlattenActionSpaceWrapper,
    FlattenObservationWrapper,
)
from mani_skill.utils.wrappers.visual_encoders import VisualEncoderWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from tests.utils import (
    LOW_MEM_SIM_CONFIG,
    MULTI_AGENT_ENV_IDS,
    OBS_MODES,
    STATIONARY_ENV_IDS,
)


@pytest.mark.gpu_sim
@pytest.mark.parametrize("env_id", STATIONARY_ENV_IDS)
@pytest.mark.parametrize("obs_mode", OBS_MODES)
def test_recordepisode_wrapper_gpu(env_id, obs_mode):
    env = gym.make(
        env_id,
        obs_mode=obs_mode,
        render_mode="rgb_array",
        max_episode_steps=10,
        num_envs=16,
        sim_config=LOW_MEM_SIM_CONFIG,
    )
    env = RecordEpisode(
        env,
        output_dir=f"videos/pytest/{env_id}-gpu",
        trajectory_name=f"test_traj_{obs_mode}",
        info_on_video=False,
        max_steps_per_video=50,
        save_trajectory=False,
    )
    env = ManiSkillVectorEnv(
        env
    )  # this is used purely to just fix the timelimit wrapper problems
    env.reset()
    action_space = env.action_space
    for _ in range(20):
        obs, rew, terminated, truncated, info = env.step(action_space.sample())
        if truncated.any():
            env.reset()
    env.close()
    del env


@pytest.mark.parametrize("env_id", STATIONARY_ENV_IDS)
@pytest.mark.parametrize("obs_mode", OBS_MODES)
def test_recordepisode_wrapper(env_id, obs_mode):
    env = gym.make(
        env_id, obs_mode=obs_mode, render_mode="rgb_array", max_episode_steps=10
    )
    env = RecordEpisode(
        env,
        output_dir=f"videos/pytest/{env_id}",
        trajectory_name=f"test_traj_{obs_mode}",
        info_on_video=True,
    )
    env.reset()
    action_space = env.action_space
    for _ in range(20):
        obs, rew, terminated, truncated, info = env.step(action_space.sample())
        if terminated or truncated:
            env.reset()
    env.close()
    del env


@pytest.mark.gpu_sim
@pytest.mark.parametrize("env_id", STATIONARY_ENV_IDS[:1])
@pytest.mark.parametrize("obs_mode", OBS_MODES[:1])
def test_recordepisode_wrapper_gpu_render_sensor(env_id, obs_mode):
    env = gym.make(
        env_id,
        obs_mode=obs_mode,
        render_mode="sensors",
        num_envs=16,
        sim_config=LOW_MEM_SIM_CONFIG,
    )
    env = RecordEpisode(
        env,
        output_dir=f"videos/pytest/{env_id}-gpu-{obs_mode}-render-sensor",
        trajectory_name=f"test_traj_{obs_mode}",
        save_trajectory=True,
        max_steps_per_video=50,
        info_on_video=False,
    )
    env = ManiSkillVectorEnv(
        env,
        max_episode_steps=10,
    )  # this is used purely to just fix the timelimit wrapper problems
    env.reset()
    action_space = env.action_space
    for _ in range(20):
        obs, rew, terminated, truncated, info = env.step(action_space.sample())
        if truncated.any():
            env.reset()
    env.close()
    del env


@pytest.mark.parametrize("env_id", STATIONARY_ENV_IDS)
@pytest.mark.parametrize("obs_mode", OBS_MODES)
def test_recordepisode_wrapper_render_sensor(env_id, obs_mode):
    env = gym.make(
        env_id, obs_mode=obs_mode, render_mode="sensors", max_episode_steps=10
    )
    env = RecordEpisode(
        env,
        output_dir=f"videos/pytest/{env_id}-{obs_mode}-render-sensor",
        trajectory_name=f"test_traj_{obs_mode}",
        info_on_video=True,
    )
    env.reset()
    action_space = env.action_space
    for _ in range(20):
        obs, rew, terminated, truncated, info = env.step(action_space.sample())
        if terminated or truncated:
            env.reset()
    env.close()
    del env


@pytest.mark.gpu_sim
@pytest.mark.parametrize("env_id", STATIONARY_ENV_IDS[:1])
@pytest.mark.parametrize("obs_mode", OBS_MODES[:1])
def test_recordepisode_wrapper_partial_reset_gpu(env_id, obs_mode):
    env = gym.make(
        env_id,
        obs_mode=obs_mode,
        render_mode="rgb_array",
        num_envs=16,
        sim_config=LOW_MEM_SIM_CONFIG,
    )
    env = RecordEpisode(
        env,
        output_dir=f"videos/pytest/{env_id}-gpu-{obs_mode}-partial-resets",
        trajectory_name=f"test_traj_{obs_mode}",
        save_trajectory=True,
        max_steps_per_video=50,
        info_on_video=False,
    )
    env = ManiSkillVectorEnv(
        env,
        max_episode_steps=10,
    )  # this is used purely to just fix the timelimit wrapper problems
    env.reset()
    action_space = env.action_space
    for i in range(20):
        obs, rew, terminated, truncated, info = env.step(action_space.sample())
        if i == 13:
            # should observe in videos (which are organized column by column order) 0, 1, 14, 15 get reset in the middle
            env.reset(options=dict(env_idx=[0, 1, 14, 15]))
    env.close()
    del env


@pytest.mark.parametrize("env_id", STATIONARY_ENV_IDS[:1])
@pytest.mark.parametrize("obs_mode", OBS_MODES[:1])
def test_recordepisode_wrapper_partial_reset(env_id, obs_mode):
    env = gym.make(
        env_id,
        obs_mode=obs_mode,
        num_envs=1,
        render_mode="rgb_array",
        sim_config=LOW_MEM_SIM_CONFIG,
    )
    env = RecordEpisode(
        env,
        output_dir=f"videos/pytest/{env_id}-{obs_mode}-partial-resets",
        trajectory_name=f"test_traj_{obs_mode}",
        save_trajectory=True,
        max_steps_per_video=50,
        info_on_video=False,
    )
    env = ManiSkillVectorEnv(
        env,
        max_episode_steps=10,
    )  # this is used purely to just fix the timelimit wrapper problems
    env.reset()
    action_space = env.action_space
    for i in range(20):
        obs, rew, terminated, truncated, info = env.step(action_space.sample())
        if i == 13:
            env.reset()
    env.close()
    del env


@pytest.mark.gpu_sim
@pytest.mark.parametrize("env_id", STATIONARY_ENV_IDS[:1])
def test_visualencoders_gpu(env_id):
    env = gym.make(
        env_id,
        obs_mode="rgbd",
        render_mode="rgb_array",
        max_episode_steps=10,
        num_envs=16,
        sim_config=LOW_MEM_SIM_CONFIG,
    )
    assert (
        "embedding" not in env.observation_space.keys()
        and "sensor_data" in env.observation_space.keys()
    )
    env = VisualEncoderWrapper(env, encoder="r3m")
    env = ManiSkillVectorEnv(
        env
    )  # this is used purely to just fix the timelimit wrapper problems
    assert env.single_observation_space["embedding"].shape == (512,)
    obs, _ = env.reset()
    assert obs["embedding"].shape == (16, 512)
    action_space = env.action_space
    for _ in range(20):
        obs, rew, terminated, truncated, info = env.step(action_space.sample())
        assert obs["embedding"].shape == (16, 512)
        if truncated.any():
            env.reset()

    env.close()
    del env


# TODO (visual encoder CPU)


@pytest.mark.gpu_sim
@pytest.mark.parametrize("env_id", STATIONARY_ENV_IDS[:1])
def test_visualencoder_flatten_gpu(env_id):
    env = gym.make(
        env_id,
        obs_mode="rgbd",
        render_mode="rgb_array",
        max_episode_steps=10,
        num_envs=16,
        sim_config=LOW_MEM_SIM_CONFIG,
    )
    env = VisualEncoderWrapper(env, encoder="r3m")
    env = FlattenObservationWrapper(env)
    env = ManiSkillVectorEnv(
        env
    )  # this is used purely to just fix the timelimit wrapper problems
    assert env.base_env.single_observation_space.shape == (540,)
    obs, _ = env.reset()
    assert obs.shape == (16, 540)
    action_space = env.action_space
    for _ in range(20):
        obs, rew, terminated, truncated, info = env.step(action_space.sample())
        assert obs.shape == (16, 540)
        if truncated.any():
            env.reset()
    env.close()
    del env


# TODO (visual encoder + flatten obs CPU)


@pytest.mark.gpu_sim
@pytest.mark.parametrize("env_id", MULTI_AGENT_ENV_IDS[:1])
def test_multi_agent_flatten_action_space_gpu(env_id):
    env = gym.make(env_id, num_envs=16, sim_config=LOW_MEM_SIM_CONFIG)
    env = FlattenActionSpaceWrapper(env)
    env.reset()
    action_space = env.action_space
    assert isinstance(action_space, gym.spaces.Box)
    assert isinstance(env.single_action_space, gym.spaces.Box)
    for _ in range(5):
        env.step(action_space.sample())
    env.close()
    del env


@pytest.mark.parametrize("env_id", MULTI_AGENT_ENV_IDS[:1])
def test_multi_agent_flatten_action_space_cpu(env_id):
    env = gym.make(env_id, num_envs=1)
    env = FlattenActionSpaceWrapper(env)
    env.reset()
    action_space = env.action_space
    assert isinstance(action_space, gym.spaces.Box)
    assert isinstance(env.single_action_space, gym.spaces.Box)
    for _ in range(5):
        env.step(action_space.sample())
    env.close()
    del env
