import gymnasium as gym
import mani_skill2.envs

env = gym.make(
    'PickCube-v0',
    robot_uid='fetch',
    # render_mode='human',
)

import numpy as np

env.reset()
while True:
    # action = env.action_space.sample()
    action = np.zeros(env.action_space.shape)
    env.step(action=action)
    env.render()
