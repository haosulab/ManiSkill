import gymnasium as gym
import pytest

import mani_skill2.envs
from mani_skill2.utils.wrappers import RecordEpisode
from mani_skill2.utils.wrappers.flatten import FlattenObservationWrapper
from mani_skill2.utils.wrappers.visual_encoders import VisualEncoderWrapper
from mani_skill2.vector.wrappers.gymnasium import ManiSkillVectorEnv
from tests.utils import ENV_IDS, OBS_MODES


@pytest.mark.gpu_sim
@pytest.mark.parametrize("env_id", ENV_IDS)
@pytest.mark.parametrize("obs_mode", OBS_MODES)
def test_recordepisode_wrapper_gpu(env_id, obs_mode):
    env = gym.make(
        env_id,
        obs_mode=obs_mode,
        render_mode="rgb_array",
        max_episode_steps=10,
        num_envs=16,
    )
    env = RecordEpisode(
        env,
        output_dir=f"videos/pytest/{env_id}-gpu",
        trajectory_name=f"test_traj_{obs_mode}",
        info_on_video=False,
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


@pytest.mark.gpu_sim
@pytest.mark.parametrize("env_id", ENV_IDS[:1])
@pytest.mark.parametrize("obs_mode", OBS_MODES[:1])
def test_recordepisode_wrapper_gpu_render_sensor(env_id, obs_mode):
    env = gym.make(
        env_id,
        obs_mode=obs_mode,
        render_mode="sensors",
        max_episode_steps=10,
        num_envs=16,
    )
    env = RecordEpisode(
        env,
        output_dir=f"videos/pytest/{env_id}-gpu-render-sensor",
        trajectory_name=f"test_traj_{obs_mode}",
        info_on_video=False,
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


@pytest.mark.gpu_sim
@pytest.mark.parametrize("env_id", [ENV_IDS[0]])
def test_visualencoders_gpu(env_id):
    env = gym.make(
        env_id,
        obs_mode="rgbd",
        render_mode="rgb_array",
        max_episode_steps=10,
        num_envs=16,
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


@pytest.mark.gpu_sim
@pytest.mark.parametrize("env_id", [ENV_IDS[0]])
def test_visualencoder_flatten_gpu(env_id):
    env = gym.make(
        env_id,
        obs_mode="rgbd",
        render_mode="rgb_array",
        max_episode_steps=10,
        num_envs=16,
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
