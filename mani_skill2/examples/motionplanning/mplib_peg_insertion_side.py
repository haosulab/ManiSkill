import gymnasium as gym
import numpy as np
import pymp
import sapien.core as sapien

from mani_skill2 import ASSET_DIR
from mani_skill2.envs.assembly.peg_insertion_side import PegInsertionSideEnv

# isort: off
from mani_skill2.examples.motionplanning.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)
from mani_skill2.examples.motionplanning.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill2.utils.wrappers import RecordEpisode
import trimesh.creation
import trimesh.sample


def main():
    env: PegInsertionSideEnv = gym.make(
        "PegInsertionSide-v0",
        obs_mode="none",
        control_mode="pd_joint_pos_vel",
        render_mode="rgb_array",
        reward_mode="dense",
        shader_dir="rt-fast",
    )
    env = RecordEpisode(
        env,
        output_dir="videos/peg_insertion_side",
        trajectory_name="trajectory",
        video_fps=60,
        info_on_video=True,
    )
    for seed in range(5, 10):
        solve(env, seed=seed, debug=False, vis=False)

    env.close()


def solve(env: PegInsertionSideEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        joint_vel_limits=0.5,
        joint_acc_limits=0.5,
    )
    env = env.unwrapped
    FINGER_LENGTH = 0.025

    obb = get_actor_obb(env.peg)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[:3, 1]
    peg_size = np.array(env.peg_half_size) * 2

    import copy

    peg_init_pose = copy.deepcopy(env.peg.pose)

    grasp_info = compute_grasp_info_by_obb(
        obb, approaching=approaching, target_closing=target_closing, depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    offset = sapien.Pose([-max(0.05, env.peg_half_size[0] / 2 + 0.01), 0, 0])
    grasp_pose = grasp_pose * (offset)

    reach_pose = grasp_pose * (sapien.Pose([0, 0, -0.05]))
    planner.move_to_pose_with_screw(reach_pose)
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()
    # align the peg with the hole
    insert_pose = env.goal_pose * peg_init_pose.inv() * grasp_pose
    offset = sapien.Pose([-0.01 - env.peg_half_size[0], 0, 0])
    pre_insert_pose = insert_pose * (offset)
    planner.move_to_pose_with_screw(pre_insert_pose)
    # refine the insertion pose
    for i in range(3):
        delta_pose = env.goal_pose * (offset) * env.peg.pose.inv()
        pre_insert_pose = delta_pose * pre_insert_pose
        planner.move_to_pose_with_screw(pre_insert_pose)

    obs, reward, terminated, truncated, info = planner.move_to_pose_with_screw(
        insert_pose * (sapien.Pose([0.05, 0, 0]))
    )
    print(info)
    planner.close()


if __name__ == "__main__":
    main()
