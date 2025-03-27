"""
Code for system ID to tune joint stiffness and damping and test out controllers compared with real robots

The strategy employed here is to sample a number of random joint positions and move the robot to each of those positions with the same actions and see
how the robot's qpos and qvel values evolve over time compared to the real world values.


To run we first save a motion profile from the real robot

python -m mani_skill.utils.calibration.system_id \
    --robot_uid so100 --real
"""
import gymnasium as gym

import mani_skill.envs
from mani_skill.envs.sim2real_env import Sim2RealEnv

if __name__ == "__main__":

    env = gym.make("Empty-v1", obs_mode="state_dict", robot_uids="so100")
    env.reset()
    real_env = gym.make("Empty-v1", obs_mode="state_dict", robot_uids="so100")
    sim2real_env = Sim2RealEnv(
        real_env,
    )
    while True:
        env.step(None)
        env.render_human()
