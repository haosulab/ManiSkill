# TODO(chichu): need to be removed
import sys
sys.path.append('/home/chichu/Documents/Sapien/ManiSkill2-Sim2Real')

import argparse
import os.path as osp
import gym
import numpy as np

from stable_baselines3 import PPO
from mani_skill2.utils.wrappers import RecordEpisode
from mani_skill2.utils.generate_sim_params import generate_sim_params
from train_eval.ppo_state import ContinuousTaskWrapper, SuccessInfoWrapper

from real_robot.agents.xarm import XArm7

class RealRobotEval():
    def __init__(self, env, model,
                 xarm_ip='192.168.1.229',
                 real_control_mode='pd_ee_delta_pos',
                 robot_action_scale=100) -> None:
        self.env = env
        self.model = model
        self.xarm_ip=xarm_ip
        self._real_control_mode = real_control_mode
        self.robot_action_scale = robot_action_scale
        self._configure_real_robot()
    
    def reset(self):
        self.real_robot.reset()
        obs = self.env.reset()

        return obs
    
    def step(self, action):
        action_real = np.concatenate([action[:3], np.expand_dims(np.array(action[-1]), axis=0)])
        self.real_robot.set_action(action_real, wait=True, action_scale=self.robot_action_scale)
        obs, action, done, info= self.env.step(action)

        return obs, action, done, info
    
    def predict(self, obs):
        action = self.model.predict(obs)
        return action

    def _configure_real_robot(self):
        """Create real robot agent"""
        self.real_robot = XArm7(
            self.xarm_ip, control_mode=self._real_control_mode,
            safety_boundary=[550, 0, 50, -600, 280, 0]
        )

def main():
    env_id = 'PickCube-v2'
    log_dir = "./logs/PPO/PickCube-v2"
    record_dir = "logs/PPO/"+env_id
    rollout_steps = 4000
    num_envs = 16
    obs_mode = "state"
    control_mode = "pd_ee_delta_pose"
    reward_mode = "dense"
    low_level_control_mode = 'position'
    motion_data_type = ['qpos', 'qvel', 'qacc', 'qf - passive_qf', 'qf']
    sim_params = generate_sim_params()

    # import real_robot.envs
    import mani_skill2.envs
    env = gym.make(
        env_id,
        obs_mode=obs_mode,
        reward_mode=reward_mode,
        control_mode=control_mode,
        low_level_control_mode=low_level_control_mode,
        motion_data_type=motion_data_type,
        sim_params = sim_params
    )

    env = SuccessInfoWrapper(env)
    env = RecordEpisode(env, record_dir, info_on_video=True, render_mode="cameras", motion_data_type=motion_data_type)

    #-----Load ppo policy-----#
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

    model_path = osp.join(log_dir, "best_model")
    # Load the saved model
    model = model.load(model_path)

    #-----Instantiate eval object-----#
    realroboteval = RealRobotEval(env=env, model=model)

    done = False
    obs = realroboteval.reset()
    while not done:
        action = realroboteval.predict(obs)[0]
        obs, action, done, info = realroboteval.step(action)

if __name__ == '__main__':
    main()