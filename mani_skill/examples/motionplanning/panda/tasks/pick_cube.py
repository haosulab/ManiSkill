import argparse
import gymnasium as gym
import numpy as np
import sapien.core as sapien
from tqdm import tqdm
import os.path as osp

from mani_skill.envs.tasks.pick_cube import PickCubeEnv
from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.utils.wrappers.record import RecordEpisode

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--obs-mode", type=str, default="none")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-c", "--control-mode", type=str)
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="can be sensors or rgb_array which only affect the video saving")
    parser.add_argument("--visualize", action="store_true", help="whether or not to open a GUI to visualize the solution live")
    parser.add_argument("--shader", default="default", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--record-dir", type=str, default="demos/motionplanning")
    return parser.parse_args()

def main(args):
    env_id = "PickCube-v1"
    env: PickCubeEnv = gym.make(
        env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode=args.render_mode,
        reward_mode="dense" if args.reward_mode is None else args.reward_mode,
    )
    env = RecordEpisode(env, output_dir=osp.join(args.record_dir, env_id), save_video=True, source_type="motionplanning", source_desc="official motion planning solution from ManiSkill contributors")
    for seed in tqdm(range(100)):
        res = solve(env, seed=seed, debug=False, vis=True if args.visualize else False)
        print(res[-1])
    env.close()


def solve(env: PickCubeEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped
    obb = get_actor_obb(env.cube)

    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp._objs[0].entity_pose.to_transformation_matrix()[:3, 1]
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_pose = sapien.Pose(env.goal_site.pose.p[0], grasp_pose.q)
    res = planner.move_to_pose_with_screw(goal_pose)

    planner.close()
    return res


if __name__ == "__main__":
    main(parse_args())
