from math import e
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from sympy import det
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.record import RecordEpisode

from mani_skill.vector.wrappers.sb3 import ManiSkillSB3VectorEnv

def main():
    ms3_vec_env = gym.make("PushCube-v1", num_envs=64)
    max_episode_steps = gym_utils.find_max_episode_steps_value(ms3_vec_env)
    vec_env = ManiSkillSB3VectorEnv(ms3_vec_env)

    model = PPO("MlpPolicy", vec_env, gamma=0.8, gae_lambda=0.9, n_steps=50, batch_size=128, n_epochs=8, verbose=1)
    model.learn(total_timesteps=500_000)
    model.save("ppo_pushcube")
    vec_env.close()
    del model # remove to demonstrate saving and loading

    model = PPO.load("ppo_pushcube")

    eval_vec_env = gym.make("PushCube-v1", num_envs=16, render_mode="rgb_array")
    eval_vec_env = RecordEpisode(eval_vec_env, output_dir="eval_videos", save_video=True, trajectory_name="eval_trajectory", max_steps_per_video=max_episode_steps)
    eval_vec_env = ManiSkillSB3VectorEnv(eval_vec_env)
    obs = eval_vec_env.reset()
    for i in range(max_episode_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = eval_vec_env.step(action)
if __name__ == "__main__":
    main()
