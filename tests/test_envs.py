import gymnasium as gym
import numpy as np
import pytest

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.envs.sapien_env import BaseEnv
from tests.utils import (
    CONTROL_MODES_STATIONARY_SINGLE_ARM,
    ENV_IDS,
    MULTI_AGENT_ENV_IDS,
    OBS_MODES,
    SINGLE_ARM_STATIONARY_ROBOTS,
    STATIONARY_ENV_IDS,
    assert_isinstance,
    assert_obs_equal,
)


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_all_envs(env_id):
    env = gym.make(env_id, obs_mode="state")
    obs, _ = env.reset()
    action_space = env.action_space
    for _ in range(5):
        obs, rew, terminated, truncated, info = env.step(action_space.sample())
    env.close()
    del env


@pytest.mark.parametrize("env_id", STATIONARY_ENV_IDS)
@pytest.mark.parametrize("obs_mode", OBS_MODES)
def test_envs_obs_modes(env_id, obs_mode):
    env = gym.make(env_id, obs_mode=obs_mode)
    obs, _ = env.reset()
    assert_isinstance(obs, np.ndarray)
    action_space = env.action_space
    for _ in range(5):
        obs, rew, terminated, truncated, info = env.step(action_space.sample())
    assert_isinstance(obs, np.ndarray)
    assert_isinstance(rew, float)
    assert_isinstance(terminated, bool)
    assert_isinstance(truncated, bool)
    assert_isinstance(info, [np.ndarray, bool, float, int])
    if obs_mode == "rgbd":
        for cam in obs["sensor_data"].keys():
            assert obs["sensor_data"][cam]["rgb"].shape == (128, 128, 3)
            assert obs["sensor_data"][cam]["depth"].shape == (128, 128, 1)
            assert obs["sensor_data"][cam]["depth"].dtype == np.uint16
    env.close()
    del env


@pytest.mark.parametrize("env_id", STATIONARY_ENV_IDS)
@pytest.mark.parametrize("control_mode", CONTROL_MODES_STATIONARY_SINGLE_ARM)
def test_env_control_modes(env_id, control_mode):
    env = gym.make(env_id, control_mode=control_mode)
    env.reset()
    action_space = env.action_space
    for _ in range(5):
        env.step(action_space.sample())
    env.close()
    del env


def test_env_seeded_reset():
    env = gym.make(ENV_IDS[0])
    obs, _ = env.reset(seed=2000)
    for _ in range(5):
        env.step(env.action_space.sample())
    new_obs, _ = env.reset(seed=2000)
    assert_obs_equal(obs, new_obs)

    env.reset()
    new_obs, _ = env.reset(seed=2000)
    assert_obs_equal(obs, new_obs)
    env.close()
    del env


def test_env_seeded_sequence_reset():
    N = 17
    env = gym.make(STATIONARY_ENV_IDS[0], max_episode_steps=5)
    obs, _ = env.reset(seed=2000)
    actions = [env.action_space.sample() for _ in range(N)]
    for i in range(N):
        first_obs, _, _, truncated, _ = env.step(actions[i])
        if truncated:
            first_obs, _ = env.reset()
    obs, _ = env.reset(seed=2000)
    for i in range(N):
        obs, _, _, truncated, _ = env.step(actions[i])
        if truncated:
            obs, _ = env.reset()
    env.close()
    assert_obs_equal(obs, first_obs)
    del env


def test_env_raise_value_error_for_nan_actions():
    env = gym.make(STATIONARY_ENV_IDS[0])
    obs, _ = env.reset(seed=2000)
    with pytest.raises(ValueError):
        env.step(env.action_space.sample() * np.nan)
    env.close()
    del env


@pytest.mark.parametrize("env_id", STATIONARY_ENV_IDS)
def test_states(env_id):
    env: BaseEnv = gym.make(env_id)
    obs, _ = env.reset(seed=1000)
    for _ in range(5):
        env.step(env.action_space.sample())
    state = env.get_state_dict()
    obs = env.get_obs()

    for _ in range(50):
        env.step(env.action_space.sample())
    env.set_state_dict(state)
    new_obs = env.get_obs()
    assert_obs_equal(obs, new_obs)
    env.close()
    del env


@pytest.mark.parametrize("env_id", STATIONARY_ENV_IDS)
@pytest.mark.parametrize("robot_uids", SINGLE_ARM_STATIONARY_ROBOTS)
def test_robots(env_id, robot_uids):
    if env_id in [
        "PandaAvoidObstacles-v0",
        "PegInsertionSide-v0",
        "PickClutterYCB-v0",
        "TurnFaucet-v0",
        "OpenCabinetDoor-v1",
        "OpenCabinetDrawer-v1",
        "PushChair-v1",
        "MoveBucket-v1",
    ]:
        pytest.skip(reason=f"Env {env_id} does not support robots other than panda")
    env = gym.make(env_id, robot_uids=robot_uids)
    env.reset()
    action_space = env.action_space
    for _ in range(5):
        env.step(action_space.sample())
    env.close()
    del env


@pytest.mark.parametrize("env_id", MULTI_AGENT_ENV_IDS)
def test_multi_agent(env_id):
    env = gym.make(env_id, num_envs=1)
    env.reset()
    action_space = env.action_space
    assert isinstance(action_space, gym.spaces.Dict)
    assert isinstance(env.unwrapped.single_action_space, gym.spaces.Dict)
    assert isinstance(env.unwrapped.agent, MultiAgent)
    for _ in range(5):
        env.step(action_space.sample())
    env.close()
    del env
