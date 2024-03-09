from collections import OrderedDict

import gymnasium as gym
import numpy as np
import sapien.core as sapien
from tqdm import tqdm

from mani_skill.envs.misc.avoid_obstacles import PandaAvoidObstaclesEnv
from mani_skill.examples.motionplanning.motionplanner import (
    CLOSED, PandaArmMotionPlanningSolver)
from mani_skill.examples.motionplanning.utils import (
    compute_grasp_info_by_obb, get_actor_obb)


def main():
    env: PandaAvoidObstaclesEnv = gym.make(
        "PandaAvoidObstacles-v0",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="sparse",
        # shader_dir="rt-fast",
    )
    for seed in tqdm(range(100)):
        res = solve(env, seed=seed, debug=False, vis=True)
        print(res[-1])
    env.close()


def solve(env: PandaAvoidObstaclesEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    orig_env = env
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        joint_vel_limits=0.4,
        joint_acc_limits=0.4,
    )
    env = env.unwrapped

    episode_config = env.episode_config
    for i, cfg in enumerate(episode_config["obstacles"]):
        planner.add_box_collision(
            np.array(cfg["half_size"]) * 2 + 0.01,
            pose=sapien.Pose(cfg["pose"][:3], cfg["pose"][3:]),
        )
    planner.add_box_collision([1, 1, 1], sapien.Pose(p=[0, 0, -0.51]))
    planner.add_box_collision(
        np.array(episode_config["wall"]["half_size"]) * 2 + 0.01,
        sapien.Pose(
            p=episode_config["wall"]["pose"][:3], q=episode_config["wall"]["pose"][3:]
        ),
    )

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    planner.gripper_state = CLOSED
    for i in range(10):
        res = planner.move_to_pose_with_RRTConnect(env.goal_pose, refine_steps=5)
        if res != -1:
            break
    planner.close()
    if res == -1:
        return OrderedDict(), 0, False, False, {"success": False, "elapsed_steps": 0}
    return res


if __name__ == "__main__":
    main()
