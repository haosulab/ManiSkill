import gymnasium as gym
import numpy as np

import mani_skill2.envs
from mani_skill2.utils.sapien_utils import to_numpy
from mani_skill2.utils.wrappers import RecordEpisode

if __name__ == "__main__":
    # , "StackCube-v0", "LiftCube-v0"
    num_envs = 200
    for env_id in ["PickSingleYCB-v1"]:  # , "StackCube-v0", "LiftCube-v0"]:
        env = gym.make(
            env_id,
            num_envs=num_envs,
            enable_shadow=True,
            robot_uid="panda",
            reward_mode="normalized_dense",
            render_mode="cameras",
            control_mode="pd_ee_delta_pos",
            sim_freq=500,
            control_freq=100,
        )
        env = RecordEpisode(
            env,
            output_dir="videos/manual_test",
            trajectory_name=f"{env_id}",
            info_on_video=False,
            video_fps=30,
            save_trajectory=False,
        )
        env.reset(seed=2)
        # env.reset(seed=1)
        done = False
        i = 0
        if num_envs == 1:
            viewer = env.render_human()
            viewer.paused = True
            env.render_human()
        while i < 50 or (i < 50000 and num_envs == 1):
            action = env.action_space.sample()
            action[:] * 0
            action[:, 2] = -1
            obs, rew, terminated, truncated, info = env.step(action)
            done = np.logical_or(to_numpy(terminated), to_numpy(truncated))
            if num_envs == 1:
                env.render_human()
            done = done.any()
            i += 1
        env.close()
