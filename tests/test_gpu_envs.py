import gymnasium as gym
import numpy as np
import pytest
import torch

from mani_skill2.agents.multi_agent import MultiAgent
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.structs.types import SimConfig
from mani_skill2.vector.wrappers.gymnasium import ManiSkillVectorEnv
from tests.utils import (
    CONTROL_MODES_STATIONARY_SINGLE_ARM,
    ENV_IDS,
    LOW_MEM_SIM_CFG,
    MULTI_AGENT_ENV_IDS,
    OBS_MODES,
    ROBOTS,
    STATIONARY_ENV_IDS,
    assert_isinstance,
    tree_map,
)


@pytest.mark.gpu_sim
@pytest.mark.parametrize("env_id", ENV_IDS)
@pytest.mark.parametrize("obs_mode", OBS_MODES)
def test_envs_obs_modes(env_id, obs_mode):
    def assert_device(x):
        assert x.device == torch.device("cuda:0")

    env = gym.make_vec(
        env_id,
        num_envs=16,
        vectorization_mode="custom",
        vector_kwargs=dict(obs_mode=obs_mode, sim_cfg=LOW_MEM_SIM_CFG),
    )
    obs, _ = env.reset()
    assert_isinstance(obs, torch.Tensor)
    tree_map(obs, lambda x: assert_device(x))
    action_space = env.action_space
    for _ in range(5):
        obs, rew, terminated, truncated, info = env.step(action_space.sample())
    assert_isinstance(obs, torch.Tensor)
    assert_isinstance(rew, torch.Tensor)
    assert_isinstance(terminated, torch.Tensor)
    assert_isinstance(truncated, torch.Tensor)
    assert_isinstance(info, [torch.Tensor])

    for obj in [rew, terminated, truncated]:
        assert_device(obj)
    tree_map(obs, lambda x: assert_device(x))
    tree_map(info, lambda x: assert_device(x))

    if obs_mode == "rgbd":
        for cam in obs["sensor_data"].keys():
            assert obs["sensor_data"][cam]["rgb"].shape == (16, 128, 128, 3)
            assert obs["sensor_data"][cam]["depth"].shape == (16, 128, 128, 1)
            assert obs["sensor_data"][cam]["depth"].dtype == torch.int16
    env.close()
    del env


@pytest.mark.gpu_sim
@pytest.mark.parametrize("env_id", STATIONARY_ENV_IDS)
@pytest.mark.parametrize("control_mode", CONTROL_MODES_STATIONARY_SINGLE_ARM)
def test_env_control_modes(env_id, control_mode):
    env = gym.make_vec(
        env_id,
        num_envs=16,
        vectorization_mode="custom",
        vector_kwargs=dict(control_mode=control_mode, sim_cfg=LOW_MEM_SIM_CFG),
    )
    env.reset()
    action_space = env.action_space
    assert action_space.shape[0] == 16
    for _ in range(5):
        env.step(action_space.sample())
    env.close()
    del env


@pytest.mark.gpu_sim
@pytest.mark.parametrize("env_id", ["PickSingleYCB-v1"])
def test_env_reconfiguration(env_id):
    env = gym.make_vec(env_id, num_envs=16, vectorization_mode="custom")
    env.reset(options=dict(reconfigure=True))
    for _ in range(5):
        env.step(env.action_space.sample())
    env.reset(options=dict(reconfigure=True))
    for _ in range(5):
        env.step(env.action_space.sample())
    env.close()
    del env


# GPU sim is not deterministic, so we do not run this test which we run for CPU sim
# def test_env_seeded_reset():
#     env = gym.make(ENV_IDS[0], num_envs=16)
#     obs, _ = env.reset(seed=2000)
#     for _ in range(5):
#         env.step(env.action_space.sample())
#     new_obs, _ = env.reset(seed=2000)
#     assert_obs_equal(obs, new_obs)

#     env.reset()
#     new_obs, _ = env.reset(seed=2000)
#     assert_obs_equal(obs, new_obs)
#     env.close()
#     del env


# def test_env_seeded_sequence_reset():
#     N = 17
#     env = gym.make(ENV_IDS[0], max_episode_steps=5)
#     obs, _ = env.reset(seed=2000)
#     actions = [env.action_space.sample() for _ in range(N)]
#     for i in range(N):
#         first_obs, _, _, truncated, _ = env.step(actions[i])
#         if truncated:
#             first_obs, _ = env.reset()
#     obs, _ = env.reset(seed=2000)
#     for i in range(N):
#         obs, _, _, truncated, _ = env.step(actions[i])
#         if truncated:
#             obs, _ = env.reset()
#     env.close()
#     assert_obs_equal(obs, first_obs)
#     del env


# def test_env_raise_value_error_for_nan_actions():
#     env = gym.make(ENV_IDS[0])
#     obs, _ = env.reset(seed=2000)
#     with pytest.raises(ValueError):
#         env.step(env.action_space.sample() * np.nan)
#     env.close()
#     del env


# @pytest.mark.parametrize("env_id", ENV_IDS)
# def test_states(env_id):
#     env: BaseEnv = gym.make(env_id, num_envs=16)
#     obs, _ = env.reset(seed=1000)
#     for _ in range(5):
#         env.step(env.action_space.sample())
#     state = env.get_state()
#     obs = env.get_obs()

#     for _ in range(50):
#         env.step(env.action_space.sample())
#     env.set_state(state)
#     new_obs = env.get_obs()
#     assert_obs_equal(obs, new_obs)
#     env.close()
#     del env


@pytest.mark.gpu_sim
@pytest.mark.parametrize("env_id", ENV_IDS)
@pytest.mark.parametrize("robot_uids", ROBOTS)
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
    env = gym.make_vec(
        env_id,
        num_envs=16,
        vectorization_mode="custom",
        vector_kwargs=dict(robot_uids=robot_uids, sim_cfg=LOW_MEM_SIM_CFG),
    )
    env.reset()
    action_space = env.action_space
    for _ in range(5):
        env.step(action_space.sample())
    env.close()
    del env


@pytest.mark.gpu_sim
@pytest.mark.parametrize("env_id", MULTI_AGENT_ENV_IDS)
def test_multi_agent(env_id):
    env = gym.make_vec(
        env_id,
        num_envs=16,
        vectorization_mode="custom",
        vector_kwargs=dict(sim_cfg=LOW_MEM_SIM_CFG),
    )
    env.reset()
    action_space = env.action_space
    assert isinstance(action_space, gym.spaces.Dict)
    assert isinstance(env.base_env.single_action_space, gym.spaces.Dict)
    assert isinstance(env.base_env.agent, MultiAgent)
    for _ in range(5):
        env.step(action_space.sample())
    env.close()
    del env


@pytest.mark.gpu_sim
@pytest.mark.parametrize("env_id", ENV_IDS[:1])
def test_partial_resets(env_id):
    env: ManiSkillVectorEnv = gym.make_vec(
        env_id,
        num_envs=16,
        vectorization_mode="custom",
        vector_kwargs=dict(sim_cfg=LOW_MEM_SIM_CFG),
    )
    obs, _ = env.reset()
    action_space = env.action_space
    for _ in range(5):
        obs, _, _, _, _ = env.step(action_space.sample())
    env_idx = torch.arange(0, 16, device=env.device)
    reset_mask = torch.zeros(16, dtype=bool, device=env.device)
    for i in [1, 3, 4, 13]:
        reset_mask[i] = True
    reset_obs, _ = env.reset(options=dict(env_idx=env_idx[reset_mask]))
    assert torch.isclose(obs[~reset_mask], reset_obs[~reset_mask]).all()
    assert not torch.isclose(
        obs[reset_mask][:, :10], reset_obs[reset_mask][:, :10]
    ).any()
    assert (env.base_env.elapsed_steps[reset_mask] == 0).all()
    assert (env.base_env.elapsed_steps[~reset_mask] == 5).all()
    env.close()
    del env


@pytest.mark.gpu_sim
@pytest.mark.parametrize("env_id", ENV_IDS[:1])
def test_timelimits(env_id):
    """Test that the vec env batches the truncated variable correctly"""
    env = gym.make_vec(
        env_id,
        num_envs=16,
        vectorization_mode="custom",
        vector_kwargs=dict(sim_cfg=LOW_MEM_SIM_CFG),
    )
    obs, _ = env.reset()
    for _ in range(50):
        obs, _, terminated, truncated, _ = env.step(None)
    assert (truncated == torch.ones(16, dtype=bool, device=env.device)).all()
    env.close()
    del env


# TODO (stao): Add test for tasks where there is no success/success and failure/no success or failure
