from functools import partial

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.vec_env import SubprocVecEnv

from mani_skill2.utils.common import flatten_dict_keys
from mani_skill2.utils.wrappers.observation import FlattenObservationWrapper
from mani_skill2.vector import make as make_vec_env
from tests.utils import ENV_IDS, ROBOTS, VENV_OBS_MODES, assert_obs_equal


@pytest.mark.skip()
def make_env(env_id, obs_mode):
    import mani_skill2.envs  # fmt: skip

    env = gym.make(env_id, obs_mode=obs_mode)
    return FlattenObservationWrapper(env)


@pytest.mark.parametrize("env_id", ["PickCube-v0", "TurnFaucet-v0"])
@pytest.mark.parametrize("obs_mode", VENV_OBS_MODES)
def test_vecenv_obs_mode(env_id, obs_mode):
    n_envs = 2

    env_fns = [partial(make_env, env_id, obs_mode=obs_mode) for _ in range(n_envs)]
    sb3_env = SubprocVecEnv(env_fns)
    ms2_env = make_vec_env(env_id, n_envs, obs_mode=obs_mode)

    np.random.seed(2022)
    sb3_env.seed(2022)
    ms2_obs, _ = ms2_env.reset(seed=2022)

    for i in range(2):
        print("Episode", i)
        sb3_obs = sb3_env.reset()
        if i != 0:
            # SB3 after env.seed(x), the first reset is a seeded reset, so we skip MS2 reset here the first time
            ms2_obs, _ = ms2_env.reset()

        assert_obs_equal(sb3_obs, ms2_obs, ignore_col_vector_shape_mismatch=True)

        for t in range(5):
            actions = ms2_env.action_space.sample()
            sb3_obs, sb3_rews, sb3_dones, sb3_infos = sb3_env.step(actions)

            (
                ms2_obs,
                ms2_rews,
                ms2_terminations,
                ms2_truncations,
                ms2_infos,
            ) = ms2_env.step(actions)
            assert_obs_equal(sb3_obs, ms2_obs, ignore_col_vector_shape_mismatch=True)
            np.testing.assert_allclose(sb3_rews, ms2_rews)
            np.testing.assert_equal(sb3_dones, ms2_terminations | ms2_truncations)
    sb3_env.close()
    ms2_env.close()
    del sb3_env
    del ms2_env


@pytest.mark.parametrize("env_id", ["PickCube-v0", "TurnFaucet-v0"])
@pytest.mark.parametrize("obs_mode", VENV_OBS_MODES)
def test_gymnasium_vecenv(env_id, obs_mode):
    n_envs = 2

    gym_env = gym.make_vec(
        env_id,
        n_envs,
        obs_mode=obs_mode,
        wrappers=[FlattenObservationWrapper],
        vectorization_mode="async",
        vector_kwargs=dict(context="forkserver"),
    )
    ms2_env = make_vec_env(env_id, n_envs, obs_mode=obs_mode)

    np.random.seed(2022)
    print("GYM")
    gym_obs, _ = gym_env.reset(seed=2022)
    print("MS2")
    ms2_obs, _ = ms2_env.reset(seed=2022)

    for i in range(2):
        print("Episode", i)
        assert_obs_equal(gym_obs, ms2_obs)

        for t in range(5):
            actions = gym_env.action_space.sample()
            (
                gym_obs,
                gym_rews,
                gym_terminations,
                gym_truncations,
                gym_infos,
            ) = gym_env.step(actions)
            (
                ms2_obs,
                ms2_rews,
                ms2_terminations,
                ms2_truncations,
                ms2_infos,
            ) = ms2_env.step(actions)

            assert_obs_equal(gym_obs, ms2_obs)
            np.testing.assert_allclose(gym_rews, ms2_rews)
            np.testing.assert_equal(gym_terminations, ms2_terminations)
            np.testing.assert_equal(gym_truncations, ms2_truncations)
    gym_env.close()
    ms2_env.close()
    del gym_env
    del ms2_env
