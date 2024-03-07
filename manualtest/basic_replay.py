import gymnasium as gym
import numpy as np
import sapien
import torch

import mani_skill.envs
from mani_skill.utils import sapien_utils
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

if __name__ == "__main__":
    # sapien.set_log_level("info")
    # , "StackCube-v1", "PickCube-v1", "PushCube-v1", "PickSingleYCB-v1", "OpenCabinet-v1"
    num_envs = 2
    for env_id in ["PickCube-v1"]:
        env = gym.make(
            env_id,
            num_envs=num_envs,
            enable_shadow=True,
            # robot_uids="fetch",
            reward_mode="normalized_dense",
            render_mode="rgb_array",
            control_mode="pd_joint_delta_pos",
            # control_mode="pd_ee_delta_pos",
            # sim_freq=100,
            # control_freq=20,
            force_use_gpu_sim=True,
            # reconfiguration_freq=1,
        )
        # env = ManiSkillVectorEnv(env)
        env = RecordEpisode(
            env,
            output_dir="videos/manual_test",
            trajectory_name="replay",
            info_on_video=False,
            video_fps=30,
            save_trajectory=True,
        )
        # env.reset(seed=4, options=dict(reconfigure=True)) # wierd qvel speeds
        env.reset(seed=52, options=dict(reconfigure=True))
        # env.reset(seed=1)

        done = False
        i = 0
        if num_envs == 1:
            viewer = env.render_human()
            viewer.paused = True
            env.render_human()
        for i in range(3):
            print("START")
            while i < 50 or (i < 50000 and num_envs == 1):
                action = env.action_space.sample()
                if len(action.shape) == 1:
                    action = action.reshape(1, -1)
                action[:] *= 0
                # action[:, -2:] *= 0
                # action[:, 6] = 1 # on fetch this controls gripper rotation
                # action[:, -3] = 1
                #
                # TODO (stao): on cpu sim, -1 here goes up, gpu sim -1 goes down?
                action[:, 1] = -1
                obs, rew, terminated, truncated, info = env.step(action)
                done = np.logical_or(
                    sapien_utils.to_numpy(terminated), sapien_utils.to_numpy(truncated)
                )
                if num_envs == 1:
                    env.render_human()
                done = done.any()
                i += 1
            env.reset(options=dict(env_idx=torch.tensor([0])))
            # env.reset(options=dict(reconfigure=True))
        env.close()
