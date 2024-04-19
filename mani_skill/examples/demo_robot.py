"""
Instantiates a empty environment with a floor, and attempts to place any given robot in there
"""

import argparse
import sys

import gymnasium as gym

import mani_skill.envs
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.wrappers.record import RecordEpisode
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--robot", type=str, required=True, help="The robot to visualize")
    parser.add_argument("--keyframe", type=str, help="A key frame to visualize the robot in. Note running this script will always print all available key frames. If no keyframe is given, this will spawn the robot at all 0.")
    parser.add_argument("--render-mode", type=str)
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("-p", "--pause", action="store_true", help="If using human render mode, auto pauses the simulation upon loading")
    parser.add_argument("--record-dir", type=str, help="Directory to record videos")
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")

    args = parser.parse_args()
    return args

def main(args):
    assert not (args.render_mode == "human" and args.record_dir is not None), "With human render mode, this script runs the viewer indefinitely so you can't record videos as well."
    env: BaseEnv = gym.make(
        "Empty-v1",
        enable_shadow=True,
        render_mode=args.render_mode,
        shader_dir=args.shader,
        robot_uids=args.robot,
        sim_backend=args.sim_backend,
    )
    env.reset(seed=0)
    if args.keyframe is not None:
        keyframe = env.unwrapped.agent.keyframes[args.keyframe]
        env.unwrapped.agent.robot.set_pose(keyframe.pose)
        env.unwrapped.agent.robot.set_qpos(keyframe.qpos)

    if args.render_mode == "human":
        viewer = env.render()
        viewer.paused = args.pause
        env.render()

    if args.render_mode == "human":
        while True:
            env.step(None)
            env.render()
    elif args.record_dir is not None:
        env = RecordEpisode(env, output_dir=args.record_dir)
        for _ in range(50):
            env.step(None)


if __name__ == "__main__":
    main(parse_args())
