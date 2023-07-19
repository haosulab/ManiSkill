import gymnasium as gym
import pytest
from tests.utils import assert_obs_equal, ENV_IDS, OBS_MODES, ROBOTS
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.wrappers import RecordEpisode


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_recordepisode_wrapper(env_id):
    env = gym.make(env_id, render_mode="cameras", max_episode_steps=20)
    env = RecordEpisode(env, output_dir=f"videos/{env_id}", info_on_video=True)
    env.reset()
    action_space = env.action_space
    for _ in range(40):
        obs, rew, terminated, truncated, info = env.step(action_space.sample())
        if terminated or truncated:
            env.reset()
    env.close()
    del env
