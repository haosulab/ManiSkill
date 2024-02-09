import gymnasium as gym
import numpy as np
import sapien.core as sapien
from tqdm import tqdm

from mani_skill2.envs.tasks.pick_cube import PickCubeEnv
from mani_skill2.examples.motionplanning.motionplanner import \
    PandaArmMotionPlanningSolver

def main():
    env: PickCubeEnv = gym.make(
        "PickCube-v1",
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


def solve(env: PickCubeEnv, seed=None, debug=False, vis=False):
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
        joint_acc_limits=0.5,
        joint_vel_limits=0.5,
    )

    env = env.unwrapped
    env.render_human()
    viewer = env._viewer
    viewer.selected_entity = planner.grasp_pose_visual._objs[0]
    while True:
        env.render_human()
        execute_current_pose = False
        if viewer.window.key_press("n"):
            execute_current_pose = True
        elif viewer.window.key_press("g"):
            planner.close_gripper()
        elif viewer.window.key_press("r"):
            planner.open_gripper()
        # TODO left, right depend on orientation really.
        elif viewer.window.key_press("down"):
            pose = planner.grasp_pose_visual.pose
            planner.grasp_pose_visual.set_pose(pose * sapien.Pose(p=[0, 0, 0.01]))
        elif viewer.window.key_press("up"):
            pose = planner.grasp_pose_visual.pose
            planner.grasp_pose_visual.set_pose(pose * sapien.Pose(p=[0, 0, -0.01]))
        elif viewer.window.key_press("right"):
            pose = planner.grasp_pose_visual.pose
            planner.grasp_pose_visual.set_pose(pose * sapien.Pose(p=[0, -0.01, 0]))
        elif viewer.window.key_press("left"):
            pose = planner.grasp_pose_visual.pose
            planner.grasp_pose_visual.set_pose(pose * sapien.Pose(p=[0, +0.01, 0]))
        if execute_current_pose:
            result = planner.move_to_pose_with_screw(planner.grasp_pose_visual.pose.sp)
            execute_current_pose = False
            print(f"Reward: {result[1]}, Info: {result[-1]}")


    # obb = get_actor_obb(env.cube._objs[0])

    # approaching = np.array([0, 0, -1])
    # target_closing = env.agent.tcp._objs[0].entity_pose.to_transformation_matrix()[:3, 1]
    # grasp_info = compute_grasp_info_by_obb(
    #     obb,
    #     approaching=approaching,
    #     target_closing=target_closing,
    #     depth=FINGER_LENGTH,
    # )
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
    main()
