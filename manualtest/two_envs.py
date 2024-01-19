import numpy as np
import sapien
import tqdm
import mani_skill2.envs
import gymnasium as gym
from mani_skill2.utils.sapien_utils import to_numpy
from mani_skill2.utils.wrappers import RecordEpisode
if __name__ == "__main__":
    sapien.physx.set_gpu_memory_config(found_lost_pairs_capacity=2**26, max_rigid_patch_count=120000)
    for env_id in ["PickCube-v0"]:
        env = gym.make(env_id, num_envs=128, obs_mode="state", enable_shadow=True, render_mode="rgb_array", control_mode="pd_joint_delta_pos", sim_freq=100, control_freq=50)
        eval_env = gym.make(env_id, num_envs=16, enable_shadow=True, render_mode="rgb_array", control_mode="pd_joint_delta_pos")
        eval_env = RecordEpisode(eval_env, output_dir="videos/manualtest", trajectory_name="twoenvs")
        env.reset(seed=0)
        for i in tqdm.tqdm(range(200)):
            obs, rew, terminated, truncated, info = env.step(env.action_space.sample())

        eval_env.reset(seed=0)
        for i in tqdm.tqdm(range(200)):
            obs, rew, terminated, truncated, info = eval_env.step(eval_env.action_space.sample())
        env.close()
        eval_env.close()