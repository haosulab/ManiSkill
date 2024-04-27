"""
Instantiates a empty environment with a floor, and attempts to place any given robot in there
"""

import argparse
import sys

import gymnasium as gym

import mani_skill.envs
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.types import SceneConfig, SimConfig
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--robot-uid", type=str, default="panda", help="The id of the robot to place in the environment")
    parser.add_argument("-k", "--keyframe", type=str, help="The name of the keyframe of the robot to display")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Seed the random actions and environment. Default is no seed",
    )
    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args = parse_args()
    env = gym.make(
        "Empty-v1",
        enable_shadow=True,
        robot_uids=args.robot_uid,
        shader_dir=args.shader,
        sim_cfg=SimConfig(sim_freq=100, scene_cfg=SceneConfig(solver_position_iterations=50)),
        render_mode="human",
    )
    env.reset(seed=0)
    env: BaseEnv = env.unwrapped
    print("Selected Robot has the following keyframes to view: ")
    print(env.agent.keyframes.keys())
    env.agent.robot.set_qpos(env.agent.robot.qpos * 0)
    if len(env.agent.keyframes) > 0:
        if args.keyframe is not None:
            kf = env.agent.keyframes[args.keyframe]
        else:
            for kf in env.agent.keyframes.values():
                break
        env.agent.robot.set_qpos(kf.qpos)
        if kf.qvel is not None:
            env.agent.robot.set_qvel(kf.qvel)
        env.agent.robot.set_pose(kf.pose)
    viewer = env.render()
    viewer.paused = True
    viewer = env.render()
    while True:
        # env.step(None)
        viewer = env.render()
