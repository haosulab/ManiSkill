import gymnasium as gym
import numpy as np
import sapien

import mani_skill2.envs
from mani_skill2.utils.sapien_utils import to_numpy
from mani_skill2.utils.wrappers import RecordEpisode
from mani_skill2.vector.wrappers.gymnasium import ManiSkillVectorEnv

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
            obs_mode="state_dict",
            reward_mode="normalized_dense",
            render_mode="rgb_array",
            control_mode="pd_joint_delta_pos",
            # sim_cfg=dict(sim_freq=100),
            # control_mode="pd_ee_delta_pos",
            # sim_freq=100,
            # control_freq=20,
            force_use_gpu_sim=True,
            # reconfiguration_freq=1,
        )
        env = RecordEpisode(
            env,
            output_dir="videos/manual_test",
            trajectory_name=f"{env_id}",
            info_on_video=False,
            video_fps=30,
            save_trajectory=True,
            max_steps_per_video=50,
        )
        env = ManiSkillVectorEnv(env)

        env.reset(seed=52, options=dict(reconfigure=True))
        for i in range(200):
            env.step(env.action_space.sample())
        env.close()

        import h5py

        data = h5py.File("videos/manual_test/PickCube-v1.h5")
        import ipdb

        ipdb.set_trace()
