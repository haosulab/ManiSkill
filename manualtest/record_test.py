import gymnasium as gym
import numpy as np
import sapien

import mani_skill.envs
from mani_skill.utils import sapien_utils
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

if __name__ == "__main__":
    # sapien.set_log_level("info")
    # , "StackCube-v1", "PickCube-v1", "PushCube-v1", "PickSingleYCB-v1", "OpenCabinet-v1"
    env_id = "PickCube-v1"
    obs_mode = "state"
    num_envs = 1
    env = gym.make(
        env_id,
        obs_mode=obs_mode,
        num_envs=1,
        render_mode="rgb_array",
    )
    # env = RecordEpisode(
    #     env,
    #     output_dir=f"videos/manual_test/{env_id}-partial-resets",
    #     trajectory_name=f"test_traj_{obs_mode}",
    #     save_trajectory=True,
    #     max_steps_per_video=50,
    #     info_on_video=False,
    # )
    env = RecordEpisode(
        env,
        output_dir="videos/manual_test",
        trajectory_name=f"{env_id}",
        info_on_video=False,
        video_fps=30,
        save_trajectory=True,
        max_steps_per_video=50,
    )
    env = ManiSkillVectorEnv(
        env,
    )  # this is used purely to just fix the timelimit wrapper problems
    env.reset()
    action_space = env.action_space
    for i in range(20):
        obs, rew, terminated, truncated, info = env.step(action_space.sample())
        if i == 13:
            env.reset()
    env.close()
    del env
