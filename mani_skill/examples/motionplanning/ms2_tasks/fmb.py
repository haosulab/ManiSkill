import gymnasium as gym
import numpy as np
import sapien.core as sapien
from tqdm import tqdm
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import qconjugate, qmult

from mani_skill.envs.tasks import FMBEnv
from mani_skill.examples.motionplanning.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.utils import (
    compute_grasp_info_by_obb, get_actor_obb)
from mani_skill.utils.wrappers.record import RecordEpisode


def main():
    env: FMBEnv = gym.make(
        "FMBEnv-v0",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="sparse",
        shader_dir="rt",
    )
    # env = RecordEpisode(env, output_dir="videos/manual_test", save_trajectory=False, save_video=True, video_fps=60)
    for seed in tqdm(range(100)):
        res = solve(env, seed=seed, debug=False, vis=True)
        print(res[-1])
    env.close()


def solve(env: FMBEnv, seed=None, debug=False, vis=False):
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
        joint_acc_limits=0.4,
        joint_vel_limits=0.4,
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped
    #
    # obb = get_actor_obb(env.bridge._objs[0])
    import trimesh.primitives

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    grasp_pose = sapien.Pose(
        p=env.bridge_grasp.pose.sp.p, q=euler2quat(np.pi, 0, np.pi / 2)
    )
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Move to reorientation box if necessary
    # -------------------------------------------------------------------------- #
    # goal_pose = sapien.Pose(env.goal_pos + [0, 0, 0.01], grasp_pose.q)
    res = planner.move_to_pose_with_screw(reach_pose)

    reach_pose = sapien.Pose(
        env.reorienting_fixture.pose.sp.p, q=grasp_pose.q
    ) * sapien.Pose([0, 0, -0.12])
    res = planner.move_to_pose_with_screw(reach_pose)

    reach_pose = sapien.Pose(
        env.reorienting_fixture.pose.sp.p, q=grasp_pose.q
    ) * sapien.Pose([0, 0, -0.12], q=euler2quat(0, -np.pi / 4, 0))
    res = planner.move_to_pose_with_screw(reach_pose)

    reach_pose = sapien.Pose(
        env.reorienting_fixture.pose.sp.p, q=grasp_pose.q
    ) * sapien.Pose([0, 0, -0.04], q=euler2quat(0, -np.pi / 4, 0))
    res = planner.move_to_pose_with_screw(reach_pose)

    planner.open_gripper()
    reach_pose = sapien.Pose(
        env.reorienting_fixture.pose.sp.p, q=grasp_pose.q
    ) * sapien.Pose([0, 0, -0.12])
    # res = planner.move_to_pose_with_screw(reach_pose)

    grasp_pose = sapien.Pose(
        p=env.bridge_grasp.pose.sp.p, q=euler2quat(np.pi, -np.pi / 4, np.pi / 2)
    )

    # qq = sapien.Pose([0.00524457, 0.00703304, -0.00323309], [2.26647e-05, 0.999048, -2.11447e-05, -0.0436269])
    reach_pose = sapien.Pose(
        env.reorienting_fixture.pose.sp.p, q=grasp_pose.q
    ) * sapien.Pose([0, 0, -0.02])
    res = planner.move_to_pose_with_screw(reach_pose)
    planner.close_gripper()
    reach_pose = sapien.Pose(
        env.reorienting_fixture.pose.sp.p, q=grasp_pose.q
    ) * sapien.Pose([0, 0, -0.3])
    res = planner.move_to_pose_with_screw(reach_pose)

    grasp_pose = sapien.Pose(
        p=env.bridge_grasp.pose.sp.p, q=euler2quat(np.pi, 0, np.pi / 2)
    )
    reach_pose = sapien.Pose(env.board.pose.sp.p, q=grasp_pose.q) * sapien.Pose(
        [0, 0, -0.3]
    )
    res = planner.move_to_pose_with_screw(reach_pose)
    rot_diff = qmult(env.bridge.pose.sp.q, qconjugate(euler2quat(0, 0, np.pi / 2)))
    reach_pose = sapien.Pose(
        env.board.pose.sp.p, q=qmult(grasp_pose.q, euler2quat(0, -np.pi / 30, 0))
    ) * sapien.Pose(
        [env.bridge.pose.sp.inv().p[0], env.bridge.pose.sp.inv().inv().p[1], -0.13]
    )
    res = planner.move_to_pose_with_screw(reach_pose, refine_steps=10)
    reach_pose = sapien.Pose(
        env.board.pose.sp.p, q=qmult(grasp_pose.q, euler2quat(0, -np.pi / 28, 0))
    ) * sapien.Pose([0, 0, -0.05])
    res = planner.move_to_pose_with_screw(reach_pose, refine_steps=10)
    planner.open_gripper()
    reach_pose = sapien.Pose(env.board.pose.sp.p, q=grasp_pose.q) * sapien.Pose(
        [0, 0, -0.2]
    )
    res = planner.move_to_pose_with_screw(reach_pose, refine_steps=10)
    res = planner.move_to_pose_with_screw(reach_pose, refine_steps=10)
    planner.close()
    return res


if __name__ == "__main__":
    main()
