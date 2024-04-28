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
    parser.add_argument("-c", "--control-mode", type=str, default="pd_joint_pos", help="The control mode to use. Note that for new robots being implemented if the _controller_configs is not implemented in the selected robot, we by default provide two default controllers, 'pd_joint_pos' and 'pd_joint_delta_pos' ")
    parser.add_argument("-k", "--keyframe", type=str, help="The name of the keyframe of the robot to display")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--keyframe-actions", action="store_true", help="Whether to use the selected keyframe to set joint targets to try and hold the robot in its position")
    parser.add_argument("--random-actions", action="store_true", help="Whether to sample random actions to control the agent. If False, no control signals are sent and it is just rendering.")
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
        obs_mode="none",
        reward_mode="none",
        enable_shadow=True,
        control_mode=args.control_mode,
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
    kf = None
    if len(env.agent.keyframes) > 0:
        if args.keyframe is not None:
            kf = env.agent.keyframes[args.keyframe]
        else:
            for kf in env.agent.keyframes.values():
                # keep the first keyframe we find
                break
        if kf.qpos is not None:
            env.agent.robot.set_qpos(kf.qpos)
        if kf.qvel is not None:
            env.agent.robot.set_qvel(kf.qvel)
        env.agent.robot.set_pose(kf.pose)
    viewer = env.render()
    viewer.paused = True
    viewer = env.render()
    while True:
        if args.random_actions:
            env.step(env.action_space.sample())
        elif args.keyframe_actions:
            assert kf is not None, "this robot has no keyframes, cannot use it to set actions"
            env.step(kf.qpos)
        viewer = env.render()
