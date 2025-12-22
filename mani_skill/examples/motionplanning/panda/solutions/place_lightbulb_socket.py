import gymnasium as gym
import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.tabletop.colosseum_v2_versions.place_lightbulb_socket import PickLightbulbPlaceSocketEnv
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb


def solve(env, debug=False,seed=None,  vis=False):
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

    env = env.unwrapped
    env.reset(seed=seed)
    obb = get_actor_obb(env.lightbulb)

    approaching = np.array([1, 0, 0])
    target_closing = np.array([0, 0, -1])
    grasp_info = compute_grasp_info_by_obb(obb, approaching, target_closing, depth=0.02)
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)

    reach_pose = grasp_pose * sapien.Pose(p=[0, 0, -0.08])
    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1:
        res = planner.move_to_pose_with_RRTConnect(reach_pose)
        if res == -1:
            return False

    planner.open_gripper()
    
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1:
        return False

    planner.close_gripper()

    lift_pose = grasp_pose * sapien.Pose(p=[0, 0, -0.15])
    res = planner.move_to_pose_with_screw(lift_pose)
    if res == -1:
        return False

    socket_pos = env.socket_pos
    above_socket_pos = np.array([socket_pos[0], socket_pos[1], socket_pos[2] + 0.12])
    above_socket_pose = sapien.Pose(
        p=above_socket_pos,
        q=euler2quat(0, np.pi / 2, 0)
    )
    res = planner.move_to_pose_with_RRTConnect(above_socket_pose)
    if res == -1:
        res = planner.move_to_pose_with_screw(above_socket_pose)
        if res == -1:
            return False

    insert_pos = np.array([socket_pos[0], socket_pos[1], socket_pos[2] + 0.02])
    insert_pose = sapien.Pose(
        p=insert_pos,
        q=euler2quat(0, np.pi / 2, 0)
    )
    res = planner.move_to_pose_with_screw(insert_pose)
    if res == -1:
        return False

    planner.open_gripper()

    retract_pose = insert_pose * sapien.Pose(p=[0, 0, -0.10])
    res = planner.move_to_pose_with_screw(retract_pose)
    if res == -1:
        return False

    planner.close()
    return True


def main():
    env = gym.make(
        "PickLightbulbPlaceSocket-v1",
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="human",
        reward_mode="dense",
    )

    for seed in range(10):
        print(f"\n{'='*50}")
        print(f"Episode {seed}")
        print(f"{'='*50}")
        
        obs, _ = env.reset(seed=seed)
        
        result = solve(env, debug=False, vis=True)
        
        if result:
            print(f"Episode {seed}: SUCCESS")
        else:
            print(f"Episode {seed}: FAILED")

    env.close()


if __name__ == "__main__":
    main()