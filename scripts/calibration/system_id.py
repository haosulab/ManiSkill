"""
Code for system ID to tune joint stiffness and damping and test out controllers compared with real robots. You are recommended to copy this code
to your own project and modify it according to your own hardware

The strategy employed here is to sample a number of random joint positions and move the robot to each of those positions with the same actions and see
how the robot's qpos and qvel values evolve over time compared to the real world values.


To run we first save a motion profile from the real robot

python -m system_id.py \
    --robot_uid so100 --real
"""
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import tyro

# from mani_skill.agents.robots.lerobot.manipulator import LeRobotRealAgent
import mani_skill.envs
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.sim2real_env import Sim2RealEnv


@dataclass
class Args:
    robot_uid: str = "so100"
    """the ID of the robot to use"""
    real: bool = True
    """whether to use the real robot"""
    seed: int = 0
    """the seed to use for the random number generator"""


def main(args: Args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ### Fill in the code for your real robot here ###
    real_agent = None  #

    # LeRobotRealAgent(real_robot)

    # env = gym.make("Empty-v1", obs_mode="state_dict", robot_uids="so100")
    # env.reset()
    sim_env = gym.make("Empty-v1", obs_mode="state_dict", robot_uids="so100")
    sim_env.reset()
    base_env: BaseEnv = sim_env.unwrapped
    # real_env = Sim2RealEnv(sim_env=sim_env, agent=real_agent)

    # In simulation we first sample some target joint positions and check that they do not cause any collisions
    # you can also update this code with your own set of joint positions to try instead
    # NUM_TARGET_JOINT_POSITIONS = 300

    # for _ in range(NUM_TARGET_JOINT_POSITIONS):
    #     passed = False
    #     while not passed:
    #         n_joints = len(base_env.agent.robot.active_joints)
    #         low = []
    #         high = []
    #         for joint in base_env.agent.robot.active_joints:
    #             low.append(joint.limits[0, 0])
    #             high.append(joint.limits[0, 1])
    #         sample_qpos = np.random.uniform(low=low, high=high, size=n_joints)
    #         base_env.agent.robot.set_qpos(sample_qpos)
    #         sim_env.step(None)
    #         for contact in base_env.scene.px.get_contacts():
    #             # ignore contacts between
    #             import ipdb; ipdb.set_trace()
    #         passed = True
    #         base_env.render_human()

    target_joint_positions = [
        [0, 2.7, 2.7, 1.0, -np.pi / 2, 0],
        [-np.pi / 2, np.pi / 2, np.pi / 2, -0.250, 0, 1.7],
        [np.pi / 3, 0.5, 0.0, -0.5, np.pi / 2, 0.3],
    ]
    # base_env.render_human().paused=True
    # base_env.render_human()

    # for target_joint_position in target_joint_positions:
    #     base_env.agent.robot.set_qpos(target_joint_position)
    #     sim_env.step(None)
    #     base_env.render_human()

    # while True:
    #     env.step(None)
    #     env.render_human()


if __name__ == "__main__":
    main(tyro.cli(Args))
