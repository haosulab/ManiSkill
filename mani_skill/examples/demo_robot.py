"""
Instantiates a empty environment with a floor, and attempts to place any given robot in there
"""

import argparse

import gymnasium as gym
import mani_skill
from mani_skill.agents.controllers.base_controller import DictController
from mani_skill.envs.sapien_env import BaseEnv
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--robot-uid", type=str, default="panda", help="The id of the robot to place in the environment")
    parser.add_argument("-b", "--sim-backend", type=str, default="auto", help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("-c", "--control-mode", type=str, default="pd_joint_pos", help="The control mode to use. Note that for new robots being implemented if the _controller_configs is not implemented in the selected robot, we by default provide two default controllers, 'pd_joint_pos' and 'pd_joint_delta_pos' ")
    parser.add_argument("-k", "--keyframe", type=str, help="The name of the keyframe of the robot to display")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--keyframe-actions", action="store_true", help="Whether to use the selected keyframe to set joint targets to try and hold the robot in its position")
    parser.add_argument("--random-actions", action="store_true", help="Whether to sample random actions to control the agent. If False, no control signals are sent and it is just rendering.")
    parser.add_argument("--none-actions", action="store_true", help="If set, then the scene and rendering will update each timestep but no joints will be controlled via code. You can use this to control the robot freely via the GUI.")
    parser.add_argument("--zero-actions", action="store_true", help="Whether to send zero actions to the robot. If False, no control signals are sent and it is just rendering.")
    parser.add_argument("--sim-freq", type=int, default=100, help="Simulation frequency")
    parser.add_argument("--control-freq", type=int, default=20, help="Control frequency")
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Seed the random actions and environment. Default is no seed",
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    env = gym.make(
        "Empty-v1",
        obs_mode="none",
        reward_mode="none",
        enable_shadow=True,
        control_mode=args.control_mode,
        robot_uids=args.robot_uid,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        render_mode="human",
        sim_config=dict(sim_freq=args.sim_freq, control_freq=args.control_freq),
        sim_backend=args.sim_backend,
    )
    env.reset(seed=0)
    env: BaseEnv = env.unwrapped
    print(f"Selected robot {args.robot_uid}. Control mode: {args.control_mode}")
    print("Selected Robot has the following keyframes to view: ")
    print(env.agent.keyframes.keys())
    env.agent.robot.set_qpos(env.agent.robot.qpos * 0)
    kf = None
    if len(env.agent.keyframes) > 0:
        kf_name = None
        if args.keyframe is not None:
            kf_name = args.keyframe
            kf = env.agent.keyframes[kf_name]
        else:
            for kf_name, kf in env.agent.keyframes.items():
                # keep the first keyframe we find
                break
        if kf.qpos is not None:
            env.agent.robot.set_qpos(kf.qpos)
            env.agent.controller.reset()
        if kf.qvel is not None:
            env.agent.robot.set_qvel(kf.qvel)
        env.agent.robot.set_pose(kf.pose)
        if kf_name is not None:
            print(f"Viewing keyframe {kf_name}")
    if env.gpu_sim_enabled:
        env.scene._gpu_apply_all()
        env.scene.px.gpu_update_articulation_kinematics()
        env.scene._gpu_fetch_all()
    viewer = env.render()
    viewer.paused = True
    viewer = env.render()
    while True:
        if args.random_actions:
            env.step(env.action_space.sample())
        elif args.none_actions:
            env.step(None)
        elif args.zero_actions:
            env.step(env.action_space.sample() * 0)
        elif args.keyframe_actions:
            assert kf is not None, "this robot has no keyframes, cannot use it to set actions"
            if isinstance(env.agent.controller, DictController):
                env.step(env.agent.controller.from_qpos(kf.qpos))
            else:
                env.step(kf.qpos)
        viewer = env.render()

if __name__ == "__main__":
    main()
