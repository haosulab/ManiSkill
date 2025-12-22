import gymnasium as gym
import numpy as np
import sapien

from mani_skill.envs.tasks.tabletop.colosseum_v2_versions.place_apple_on_plate import PlaceAppleOnPlateEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb


def main():
    env: PlaceAppleOnPlateEnv = gym.make(
        "PlaceAppleOnPlate-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
    )
    for seed in range(100):
        res = solve(env, seed=seed, debug=True, vis=True)
        print(res)
    env.close()


def solve(env: PlaceAppleOnPlateEnv, seed=None, debug=False, vis=False):
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
        joint_vel_limits=0.75,
        joint_acc_limits=0.75,
    )

    env = env.unwrapped
    FINGER_LENGTH = 0.025
    obb = get_actor_obb(env.apple)

    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
    grasp_pose = grasp_pose * sapien.Pose([0, 0, 0])

    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.1])
    res = planner.move_to_pose_with_RRTStar(reach_pose)
    if res == -1:
        return res

    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1:
        return res
    planner.close_gripper()

    lift_pose = grasp_pose * sapien.Pose([0, 0, -0.15])
    res = planner.move_to_pose_with_screw(lift_pose)
    if res == -1:
        return res

    plate_pos = env.plate.pose.p.cpu().numpy()[0]
    
    pre_place_pose = sapien.Pose(
        p=[plate_pos[0], plate_pos[1], plate_pos[2] + 0.20],
        q=lift_pose.q
    )
    res = planner.move_to_pose_with_RRTStar(pre_place_pose)
    if res == -1:
        return res

    PLATE_HEIGHT = env.PLATE_HEIGHT
    APPLE_RADIUS = env.APPLE_RADIUS
    place_z = PLATE_HEIGHT + APPLE_RADIUS + 0.01
    
    place_pose = sapien.Pose(
        p=[plate_pos[0], plate_pos[1], place_z],
        q=pre_place_pose.q
    )
    res = planner.move_to_pose_with_screw(place_pose)
    if res == -1:
        return res

    planner.open_gripper()

    res = planner.move_to_pose_with_screw(pre_place_pose)
    if res == -1:
        return res

    planner.close()

    return res


if __name__ == "__main__":
    main()