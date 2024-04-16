import gymnasium as gym
import numpy as np
import sapien

import mani_skill.envs
from mani_skill.utils.wrappers.record import RecordEpisode

env = gym.make(
    "Empty-v1", robot_uids="humanoid", render_mode="rgb_array", shader_dir="rt-med"
)
env = RecordEpisode(env, output_dir="videos")
env.reset(seed=0)
env.unwrapped.agent.robot.set_pose(sapien.Pose(p=[0, 0, 1]))
n = len(env.unwrapped.agent.robot.qpos)
# env.unwrapped.agent.robot.qpos += np.random.rand((n))
viewer = env.render_human()
viewer.paused = True
for i in range(50):
    env.step(env.action_space.sample())
    env.render_human()
env.close()
