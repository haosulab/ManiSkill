from dataclasses import dataclass

import gymnasium as gym
import torch
import tyro

from examples.baselines.ppo.ppo_rgb import Agent

# ManiSkill specific imports
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode


# TODO (xhin): suppport sim_env arg, makes default value setting much easier
# TODO (xhin): support for RecordEpisode wrapper to log real world evals
@dataclass
class Args:
    real_env_id: str = "RealGrabCube-v1"
    """environment for evalulation"""
    robot_yaml_path: str = ""
    keyframe_id: str = None
    """robot keyframe for task initial robot qpos"""
    control_freq: int = 5
    control_mode: str = "pd_joint_delta_pos"
    checkpoint: str = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    num_eval_steps: int = 50
    """maximum steps agent can take"""
    debug: bool = False
    """toggle printing state-based obs at every state, single step at a time"""


if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device("cpu")

    # make eval environment
    eval_envs = gym.make(
        args.real_env_id,
        robot_yaml_path=args.robot_yaml_path,
        keyframe_id=args.keyframe_id,
        control_freq=args.control_freq,
        control_mode=args.control_mode,
        control_timing=not args.debug,
    )
    eval_envs = FlattenRGBDObservationWrapper(
        eval_envs, rgb=True, depth=False, state=True
    )
    eval_obs, _ = eval_envs.reset()

    # recreate sim ppo_rgb.py trained agent
    agent = Agent(eval_envs, sample_obs=eval_obs).to(device)
    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))
    else:
        print("Warning: Evaluating untrained agent")

    for step in range(args.num_eval_steps):
        with torch.no_grad():
            if args.debug:
                print("Eval step", step + 1)
                a, _, _, v = agent.get_action_and_value(eval_obs)
                print("action", a)
                print("value", v)
                obs = eval_envs.get_obs()
                print("Agent Obs:")
                agent_obs = obs["agent"]
                for value in agent_obs:
                    print(value, agent_obs[value])
                print("Extra Obs:")
                extra_obs = obs["extra"]
                for value in extra_obs:
                    print(value, extra_obs[value])
                input("Press Enter for next action")
                print()
            (
                eval_obs,
                eval_rew,
                eval_terminations,
                eval_truncations,
                eval_infos,
            ) = eval_envs.step(agent.get_action(eval_obs, deterministic=True))
