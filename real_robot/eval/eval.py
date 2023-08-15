import sys
sys.path.append('/home/chichu/Documents/Sapien/ManiSkill2-Sim2Real')

import argparse
import os.path as osp
import gym
import numpy as np

from stable_baselines3 import PPO

env_id = 'PickCubeRealXArm7-v0'
log_dir = "./logs/PickCube-v1"
rollout_steps = 4000
num_envs = 16
obs_mode = "state"
control_mode = "pd_ee_delta_pose"
reward_mode = "dense"

import real_robot.envs
env = gym.make(
    env_id,
    obs_mode=obs_mode,
    reward_mode=reward_mode,
    control_mode=control_mode,
)

# Define the policy configuration and algorithm configuration
policy_kwargs = dict(net_arch=[256, 256])
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    n_steps=rollout_steps // num_envs,
    batch_size=400,
    gamma=0.8,     # default = 0.85
    gae_lambda=0.9,
    n_epochs=20,
    tensorboard_log=log_dir,
    target_kl=0.1,
)

model_path = osp.join(log_dir, "latest_model")
# Load the saved model
model = model.load(model_path)