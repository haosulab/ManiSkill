import gymnasium as gym
import numpy as np
import sapien

import mani_skill.envs
from mani_skill.utils import sapien_utils
from mani_skill.utils.wrappers import RecordEpisode

# sapien.set_log_level("info")
if __name__ == "__main__":
    # sapien.set_log_level("info")
    # , "StackCube-v1", "PickCube-v1", "PushCube-v1", "PickSingleYCB-v1", "OpenCabinet-v1"
    num_envs = 32
    for env_id in ["PickCube-v1"]:

        for i in range(100):
            env = gym.make(
                env_id,
                num_envs=num_envs,
                enable_shadow=False,
                reward_mode="normalized_dense",
                render_mode="rgb_array",
                control_mode="pd_joint_delta_pos",
                force_use_gpu_sim=True,
            )
            env.reset(seed=52, options=dict(reconfigure=True))
            print(i)
            while i < 50:
                action = env.action_space.sample()
                if len(action.shape) == 1:
                    action = action.reshape(1, -1)
                obs, rew, terminated, truncated, info = env.step(action)
                done = np.logical_or(
                    sapien_utils.to_numpy(terminated), sapien_utils.to_numpy(truncated)
                )
                if num_envs == 1:
                    env.render_human()
                done = done.any()
                i += 1
            env.reset()
            # env.reset(options=dict(reconfigure=True))
            env.close()
            # import time
            # import gc

            # del env
            # print("CLEAN UP?")
            # # time.sleep(2)
            # import ipdb;ipdb.set_trace()
            # print("CLEAN UP")
