"""
Instantiates a empty environment with a floor, and attempts to place any given robot in there
"""

from dataclasses import dataclass
from typing import Annotated, Optional
import tyro
import gymnasium as gym
import mani_skill
from mani_skill.agents.controllers.base_controller import DictController
from mani_skill.envs.sapien_env import BaseEnv
@dataclass
class Args:
    robot_uid: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "panda"
    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    control_mode: Annotated[str, tyro.conf.arg(aliases=["-c"])] = "pd_joint_pos"
    keyframe: Annotated[Optional[str], tyro.conf.arg(aliases=["-k"])] = None
    shader: str = "default"
    keyframe_actions: bool = False
    random_actions: bool = False
    none_actions: bool = False
    zero_actions: bool = False
    sim_freq: int = 100
    control_freq: int = 20
    seed: Annotated[Optional[int], tyro.conf.arg(aliases=["-s"])] = None

def main(args: Args):
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
    main(tyro.cli(Args))
