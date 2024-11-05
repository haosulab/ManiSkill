import gymnasium as gym
import numpy as np
import pytest
import torch

from mani_skill.envs.sapien_env import BaseEnv
from tests.utils import LOW_MEM_SIM_CONFIG, assert_obs_equal


@pytest.mark.gpu_sim
def test_raw_sim_states():
    # Test sim state get and set works for environment without overriden get_state_dict functions
    env = gym.make(
        "PickCube-v1", num_envs=16, obs_mode="state_dict", sim_config=LOW_MEM_SIM_CONFIG
    )
    base_env: BaseEnv = env.unwrapped
    obs1, _ = env.reset()
    state_dict = base_env.get_state_dict()
    assert isinstance(state_dict, dict)
    assert state_dict["actors"]["cube"].shape == (16, 13)
    assert state_dict["actors"]["goal_site"].shape == (16, 13)
    assert state_dict["articulations"]["panda"].shape == (16, 13 + 9 * 2)
    for i in range(5):
        env.step(env.action_space.sample())
    base_env.set_state_dict(state_dict)
    set_obs = base_env.get_obs()
    assert_obs_equal(obs1, set_obs)
    for i in range(5):
        env.step(env.action_space.sample())
    state = base_env.get_state()
    obs1 = base_env.get_obs()
    assert state.shape == (16, 13 * 3 + 13 + 9 * 2)
    for i in range(5):
        env.step(env.action_space.sample())
    base_env.set_state(state)
    set_obs = base_env.get_obs()
    assert_obs_equal(obs1, set_obs)


@pytest.mark.gpu_sim
def test_raw_heterogeneous_actor_sim_states():
    # Test sim state get and set works for environment without overriden get_state_dict functions
    env = gym.make(
        "PegInsertionSide-v1",
        num_envs=16,
        obs_mode="state_dict",
        sim_config=LOW_MEM_SIM_CONFIG,
    )
    base_env: BaseEnv = env.unwrapped
    obs1, _ = env.reset()
    state_dict = base_env.get_state_dict()
    assert isinstance(state_dict, dict)
    assert state_dict["actors"]["peg"].shape == (16, 13)
    assert state_dict["actors"]["box_with_hole"].shape == (16, 13)
    assert state_dict["articulations"]["panda_wristcam"].shape == (16, 13 + 9 * 2)
    for i in range(5):
        env.step(env.action_space.sample())
    base_env.set_state_dict(state_dict)
    set_obs = base_env.get_obs()
    assert_obs_equal(obs1, set_obs)
    for i in range(5):
        env.step(env.action_space.sample())
    state = base_env.get_state()
    obs1 = base_env.get_obs()
    assert state.shape == (16, 13 * 3 + 13 + 9 * 2)
    for i in range(5):
        env.step(env.action_space.sample())
    base_env.set_state(state)
    set_obs = base_env.get_obs()
    assert_obs_equal(obs1, set_obs)


@pytest.mark.gpu_sim
def test_raw_heterogeneous_articulations_sim_states():
    # Test sim state get and set works for environment without overriden get_state_dict functions
    env = gym.make(
        "OpenCabinetDrawer-v1",
        num_envs=16,
        obs_mode="state_dict",
        sim_config=LOW_MEM_SIM_CONFIG,
    )
    base_env: BaseEnv = env.unwrapped
    obs1, _ = env.reset()
    state_dict = base_env.get_state_dict()
    assert isinstance(state_dict, dict)
    max_dof = base_env.cabinet.max_dof
    assert state_dict["articulations"]["cabinet"].shape == (16, 13 + max_dof * 2)
    assert state_dict["articulations"]["fetch"].shape == (16, 13 + 15 * 2)
    for i in range(5):
        env.step(env.action_space.sample())
    base_env.set_state_dict(state_dict)
    set_obs = base_env.get_obs()
    assert_obs_equal(obs1, set_obs)
    for i in range(5):
        env.step(env.action_space.sample())
    state = base_env.get_state()
    obs1 = base_env.get_obs()
    assert state.shape == (16, 13 + 13 + max_dof * 2 + 13 + 15 * 2)
    for i in range(5):
        env.step(env.action_space.sample())
    base_env.set_state(state)
    set_obs = base_env.get_obs()
    assert_obs_equal(obs1, set_obs)
