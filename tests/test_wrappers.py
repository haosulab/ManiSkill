import gymnasium as gym
import pytest

from mani_skill2.utils.wrappers import RecordEpisode
from tests.utils import ENV_IDS, OBS_MODES


@pytest.mark.parametrize("env_id", ENV_IDS)
@pytest.mark.parametrize("obs_mode", OBS_MODES)
def test_recordepisode_wrapper(env_id, obs_mode):
    env = gym.make(
        env_id, obs_mode=obs_mode, render_mode="cameras", max_episode_steps=20
    )
    env = RecordEpisode(env, output_dir=f"videos/{env_id}", info_on_video=True)
    env.reset()
    action_space = env.action_space
    for _ in range(40):
        obs, rew, terminated, truncated, info = env.step(action_space.sample())
        if terminated or truncated:
            env.reset()
    env.close()
    del env
