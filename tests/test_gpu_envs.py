import gymnasium as gym
import numpy as np
import pytest
import torch

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.types import SimConfig
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from tests.utils import (
    CONTROL_MODES_STATIONARY_SINGLE_ARM,
    ENV_IDS,
    LOW_MEM_SIM_CFG,
    MULTI_AGENT_ENV_IDS,
    OBS_MODES,
    SINGLE_ARM_STATIONARY_ROBOTS,
    STATIONARY_ENV_IDS,
    assert_isinstance,
    assert_obs_equal,
    tree_map,
)


@pytest.mark.gpu_sim
@pytest.mark.parametrize("env_id", ENV_IDS)
def test_all_envs(env_id):
    sim_cfg = dict()
    if "Scene" not in env_id:
        sim_cfg = LOW_MEM_SIM_CFG
    env = gym.make(env_id, num_envs=16, obs_mode="state", sim_cfg=sim_cfg)
    obs, _ = env.reset()
    action_space = env.action_space
    for _ in range(5):
        obs, rew, terminated, truncated, info = env.step(action_space.sample())
    env.close()
    del env


@pytest.mark.gpu_sim
@pytest.mark.parametrize("env_id", STATIONARY_ENV_IDS)
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


@pytest.mark.gpu_sim
def test_raw_sim_states():
    # Test sim state get and set works for environment without overriden get_state_dict functions
    env = gym.make(
        "PickCube-v1", num_envs=16, obs_mode="state_dict", sim_cfg=LOW_MEM_SIM_CFG
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
@pytest.mark.parametrize("env_id", STATIONARY_ENV_IDS[:1])
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
    assert torch.isclose(obs[~reset_mask], reset_obs[~reset_mask], atol=1e-4).all()
    assert not torch.isclose(
        obs[reset_mask][:, :10], reset_obs[reset_mask][:, :10]
    ).any()
    assert (env.base_env.elapsed_steps[reset_mask] == 0).all()
    assert (env.base_env.elapsed_steps[~reset_mask] == 5).all()
    env.close()
    del env


@pytest.mark.gpu_sim
@pytest.mark.parametrize("env_id", STATIONARY_ENV_IDS[:1])
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


@pytest.mark.gpu_sim
@pytest.mark.parametrize("env_id", ["PickCube-v1"])
def test_hidden_objs(env_id):
    env: ManiSkillVectorEnv = gym.make_vec(
        env_id, num_envs=16, vectorization_mode="custom"
    )
    obs, _ = env.reset()
    hide_obj = env.unwrapped._hidden_objects[0]

    def test_fn():
        # note that we use torch.is_close instead of checking exact match as data can change a tiny amount (1e-7 range ish)
        # when applying then fetching new poses

        # for PickCube, this is env.goal_site
        raw_pose = hide_obj.pose.raw_pose.clone()
        p = hide_obj.pose.p.clone()
        q = hide_obj.pose.q.clone()
        linvel = hide_obj.linear_velocity.clone()
        angvel = hide_obj.angular_velocity.clone()
        # 1. check relevant hidden properties are active
        assert hide_obj.hidden
        assert hasattr(hide_obj, "before_hide_pose")

        # 2. check state data for new pos is not too low or high
        assert (
            hide_obj.px.cuda_rigid_body_data.torch()[
                hide_obj._body_data_index, :7
            ].clone()[..., :3]
            > 1e3
        ).all()
        assert (
            hide_obj.px.cuda_rigid_body_data.torch()[
                hide_obj._body_data_index, :7
            ].clone()[..., :3]
            < 1e6
        ).all()

        # 3. check that linvel and angvel same as before
        assert (hide_obj.linear_velocity == linvel).all()
        assert (hide_obj.angular_velocity == angvel).all()

        # 4. Check data stored in buffer has same q but different p
        assert (
            hide_obj.px.cuda_rigid_body_data.torch()[
                hide_obj._body_data_index, :7
            ].clone()[..., :3]
            != p
        ).all()
        assert (
            hide_obj.px.cuda_rigid_body_data.torch()[
                hide_obj._body_data_index, :7
            ].clone()[..., 3:]
            == q
        ).all()

        # 5. Check data stored in before_hide_pose has same q and p
        assert (hide_obj.before_hide_pose[..., :3] == p).all()
        assert (hide_obj.before_hide_pose[..., 3:] == q).all()

        # 6. check that direct calls to raw_pose, pos, and rot same as before
        #       (should return `before_hide_pose`)
        assert (hide_obj.pose.raw_pose == raw_pose).all()
        assert (hide_obj.pose.p == p).all()
        assert (hide_obj.pose.q == q).all()
        assert (hide_obj.pose.raw_pose == hide_obj.before_hide_pose).all()

        # show_visual tests
        hide_obj.show_visual()

        # 1. check relevant hidden properties are active
        assert not hide_obj.hidden

        # 2. check that qvel, linvel, angvel same as before
        assert (hide_obj.linear_velocity == linvel).all()
        assert (hide_obj.angular_velocity == angvel).all()

        # 3. check gpu buffer goes back to normal
        assert torch.isclose(
            hide_obj.px.cuda_rigid_body_data.torch()[
                hide_obj._body_data_index, :7
            ].clone()[..., :3],
            p,
        ).all()
        assert torch.isclose(
            hide_obj.px.cuda_rigid_body_data.torch()[
                hide_obj._body_data_index, :7
            ].clone()[..., 3:],
            q,
        ).all()

        # 4. check that direct calls to raw_pose, pos, and rot same as before
        assert torch.isclose(hide_obj.pose.raw_pose, raw_pose).all()
        assert torch.isclose(hide_obj.pose.p, p).all()
        assert torch.isclose(hide_obj.pose.q, q).all()

    # Test after reset
    hide_obj.hide_visual()
    test_fn()
    # Test after partial resets
    hide_obj.hide_visual()
    env_idx = torch.arange(16, dtype=int, device=env.device)
    reset_mask = torch.zeros(16, dtype=bool, device=env.device)
    for i in [1, 3, 7, 11]:
        reset_mask[i] = True
    env.reset(options=dict(env_idx=env_idx[reset_mask]))
    test_fn()
    env.close()
    del env


# TODO (stao): Add test for tasks where there is no success/success and failure/no success or failure
