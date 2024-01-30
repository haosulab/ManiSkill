import gymnasium as gym
import pytest

import mani_skill2.envs
from mani_skill2.utils.wrappers import RecordEpisode
from mani_skill2.vector.wrappers.gymnasium import ManiSkillVectorEnv
from tests.utils import ENV_IDS, OBS_MODES


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
