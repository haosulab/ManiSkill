import gymnasium as gym
import pytest
from tests.utils import assert_obs_equal, ENV_IDS, OBS_MODES, ROBOTS
from mani_skill2.envs.sapien_env import BaseEnv
import numpy as np


@pytest.mark.parametrize("env_id", ENV_IDS)
@pytest.mark.parametrize("obs_mode", OBS_MODES)
def test_envs(env_id, obs_mode):
    env = gym.make(env_id, obs_mode=obs_mode)
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


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_states(env_id):
    env: BaseEnv = gym.make(env_id)
    obs, _ = env.reset(seed=1000)
    for _ in range(5):
        env.step(env.action_space.sample())
    state = env.get_state()
    obs = env.get_obs()

    for _ in range(50):
        env.step(env.action_space.sample())
    env.set_state(state)
    new_obs = env.get_obs()
    assert_obs_equal(obs, new_obs)
    env.close()
    del env


@pytest.mark.parametrize("env_id", ENV_IDS)
@pytest.mark.parametrize("robot", ROBOTS)
@pytest.mark.skip(reason="xmate robot not added yet")
def test_robots(env_id, robot):
    env = gym.make(env_id, robot=robot)
    env.reset()
    action_space = env.action_space
    for _ in range(5):
        env.step(action_space.sample())
    env.close()
    del env
