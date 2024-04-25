import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.vec_env import SubprocVecEnv

import mani_skill.envs
from tests.utils import VENV_OBS_MODES


@pytest.mark.parametrize("env_id", ["PickCube-v1"])
@pytest.mark.parametrize("obs_mode", VENV_OBS_MODES)
def test_gymnasium_cpu_vecenv(env_id, obs_mode):
    n_envs = 2
    gym_env = gym.make_vec(
        env_id,
        n_envs,
        obs_mode=obs_mode,
        # wrappers=[FlattenObservationWrapper],
        vectorization_mode="sync",
    )
    np.random.seed(2022)
    gym_obs, _ = gym_env.reset(seed=2022)

    for i in range(2):
        gym_env.reset()
        for t in range(5):
            actions = gym_env.action_space.sample()
            (
                gym_obs,
                gym_rews,
                gym_terminations,
                gym_truncations,
                gym_infos,
            ) = gym_env.step(actions)

    gym_env.close()
    del gym_env
