from pathlib import Path

import gymnasium as gym
import numpy as np
import sapien.core as sapien
from tqdm import tqdm
from transforms3d.euler import euler2quat

from mani_skill import ASSET_DIR
from mani_skill.envs.pick_and_place.pick_clutter import PickClutterEnv
from mani_skill.examples.motionplanning.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.utils import (
    compute_grasp_info_by_obb, get_actor_obb)


def main():
    env: PickClutterEnv = gym.make(
        "PickClutterYCB-v0",
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


def solve(env: PickClutterEnv, seed=None, debug=False, vis=False):
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
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        joint_vel_limits=0.5,
        joint_acc_limits=0.5,
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped
    obb = get_actor_obb(env.obj)

    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.entity_pose.to_transformation_matrix()[:3, 1]
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    orig_grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    # -------------------------------------------------------------------------- #
    # Search grasp poses
    # -------------------------------------------------------------------------- #
    def search_grasp_poses(grasp_poses, n=None):
        rng = np.random.RandomState(seed)
        inds = rng.permutation(len(grasp_poses))[:n]
        init_qpos = env.agent.robot.get_qpos()
        for grasp_pose in grasp_poses[inds]:
            p = grasp_pose[:3]
            q = grasp_pose[3:7]
            w = grasp_pose[7]
            grasp_pose = env.obj.pose * sapien.Pose(p, q)

            # Symmetry
            grasp_closing = grasp_pose.to_transformation_matrix()[:3, 1]
            if (grasp_closing @ target_closing) < 0:
                grasp_pose = grasp_pose * sapien.Pose(q=euler2quat(0, 0, np.pi))

            res = planner.move_to_pose_with_screw(grasp_pose, dry_run=True)
            # If plan succeeded and is not too long, proceed
            if res != -1 and len(res["position"]) < 200:
                return grasp_pose, w, res
        return None, None

    model_id = env.obj.name
    grasp_poses_path = (
        ASSET_DIR
        / "mani_skill2_ycb/grasp_poses_info_pick_v0_panda_v2"
        / f"{model_id}.npy"
    )
    if not grasp_poses_path.exists():
        grasp_poses_path = None

    # -------------------------------------------------------------------------- #
    # Compute usable grasp pose and reach
    # -------------------------------------------------------------------------- #
    grasp_pose2 = None
    if grasp_poses_path is not None:
        grasp_poses = np.load(grasp_poses_path)
        grasp_pose2, w, res = search_grasp_poses(grasp_poses, 512)
        if grasp_pose2 is not None:
            grasp_pose = grasp_pose2
            # hardcode
            CLEARANCE = 0.002
            OPEN_GRIPPER_POS = ((w + 0.01 + CLEARANCE) / 0.05) * 2 - 1
            planner.gripper_state = OPEN_GRIPPER_POS
            reach_pose = grasp_pose * (sapien.Pose([0, 0, -0.05]))
            planner.follow_path(res)

    if grasp_pose2 is None:
        grasp_pose = orig_grasp_pose
        reach_pose = grasp_pose * (sapien.Pose([0, 0, -0.05]))
        planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Move to goal
    # -------------------------------------------------------------------------- #
    # NOTE(jigu): The goal position is defined by center of mass.
    offset = env.goal_pos - env.obj_pose.p
    goal_pose = sapien.Pose(grasp_pose.p + offset, grasp_pose.q)
    res = planner.move_to_pose_with_screw(goal_pose)
    if res == -1:
        print("Try to search goal pose")
        center_pose = sapien.Pose(p=env.goal_pos)
        rng = np.random.RandomState(seed)
        for _ in range(20):
            center_pose = sapien.Pose(p=env.goal_pos)
            q = euler2quat(
                0, rng.uniform(-np.pi / 2, 0), rng.uniform(-np.pi / 3, np.pi / 3)
            )
            delta_pose = sapien.Pose(q=q)
            goal_pose = center_pose * delta_pose * center_pose.inv() * goal_pose
            res = planner.move_to_pose_with_screw(goal_pose)
            if res != -1:
                break
        else:
            print("Fail to find a goal pose")

    # refine a bit more
    res = planner.move_to_pose_with_screw(goal_pose)

    planner.close()
    return res


if __name__ == "__main__":
    main()
