import numpy as np
import sapien
import torch
from time import sleep

from mani_skill.envs.tasks.tabletop.pour_sphere import PourSphereEnv
from transforms3d.quaternions import axangle2quat, qmult

from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


class SequentialSolver:

    def __init__(self, planner: PandaArmMotionPlanningSolver, env: PourSphereEnv):
        pass


    def approach_cup1(self):
        pass
        
    
    def move_cup1_grasp_pose(self):
        pass

    def grasp_cup1(self):
        pass

    def raise_cup1(self):
        pass

    def move_above_cup2(self):
        pass


def solve(env: PourSphereEnv, seed=None, debug=False, vis=False):
    """Grasp cup and tilt it over target cup using joint 7 rotation."""
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in ["pd_joint_pos", "pd_joint_pos_vel"]

    robot_base_pose_batched = env.unwrapped.agent.robot.pose
    robot_base_pose = sapien.Pose(
        p=robot_base_pose_batched.p[0].cpu().numpy(),
        q=robot_base_pose_batched.q[0].cpu().numpy()
    )

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=robot_base_pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    env_sim = env.unwrapped
    cup1_pos = env_sim.cup1.pose.p[0].cpu().numpy()
    cup2_pos = env_sim.cup2.pose.p[0].cpu().numpy()
    print(f"cup1_pos: {cup1_pos}")
    print(f"cup2_pos: {cup2_pos}")

    # Step 1: Reach cup
    reach_pose = cup1_pos.copy()
    reach_pose[2] += 0.3
    closing = np.array([0, -1, 0])
    approaching = np.array([1, 0, 0])
    reach_pose = env_sim.agent.build_grasp_pose(approaching, closing, reach_pose)


    planner.gripper_state = planner.OPEN
    result = planner.move_to_pose_with_screw(reach_pose)
    print(f"planner.move_to_pose_with_screw(reach_pose) -> {result}")
    if not result or result == -1:
        print(f"⚠ Screw plan FAILED at Step 1 (reach)")
        print(f"  Target pose: p={reach_pose.p}, q={reach_pose.q}")
        sleep(100)
        result = planner.move_to_pose_with_RRTConnect(reach_pose)
        if not result or result == -1:
            print(f"⚠ RRT plan FAILED at Step 1 (reach)")
            planner.close()
            return result

    # Step 2: Move to grasp pose
    # Build grasp pose for cup1 (side approach)
    grasp_center = cup1_pos.copy()
    # approaching = np.array([1, 0, 0])
    # closing = np.array([0, 1, 0])
# Pose([0.05, -0.15, 0.15], [0.707107, 0, 0.707107, 0])
# T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
#     [ -0.0000000,  0.0000000, -1.0000000;
    #    0.0000000,  1.0000000,  0.0000000;
    #    1.0000000,  0.0000000, -0.0000000 ]
    closing = np.array([0, -1, 0])
    approaching = np.array([1, 0, 0])
    # approaching = np.array([1, 0, 0])
    # closing = np.array([0, -1, 0]) # ROTATION CORRECT, but __
    grasp_pose = env_sim.agent.build_grasp_pose(approaching, closing, grasp_center)

    result = planner.move_to_pose_with_screw(grasp_pose)
    if not result or result == -1:
        print(f"⚠ Screw plan FAILED at Step 2 (grasp)")
        print(f"  Target pose: p={grasp_pose.p}, q={grasp_pose.q}")
        result = planner.move_to_pose_with_RRTConnect(grasp_pose)
        if not result or result == -1:
            print(f"⚠ RRT plan FAILED at Step 2 (grasp)")
            planner.close()
            return result

    # Step 3: Close gripper
    planner.close_gripper(t=120)
    for _ in range(30):
        env_sim.scene.step()

    # Step 4: Lift cup
    current_tcp_pose = env_sim.agent.tcp.pose
    current_tcp_p = current_tcp_pose.p[0].cpu().numpy()
    current_tcp_q = current_tcp_pose.q[0].cpu().numpy()
    lift_pose = sapien.Pose(p=current_tcp_p + np.array([0, 0, 0.20]), q=current_tcp_q)
    result = planner.move_to_pose_with_RRTConnect(lift_pose)
    if not result or result == -1:
        planner.close()
        return result

    print("cup lifted")

    cup1_pos = env_sim.cup1.pose.p[0].cpu().numpy()
    cup2_pos = env_sim.cup2.pose.p[0].cpu().numpy()
    tcp_pose = env_sim.agent.tcp.pose

    # Step 4 - translate to on top of cup2 (only along the x and y plane). Do not change the orientation
    for i in range(10):
        pct = (i+1) / 10
        p_target = (1 - pct)*cup1_pos + pct*cup2_pos
        above_cup2_target_pose = sapien.Pose(p=p_target + np.array([0, 0, 0.2]), q=tcp_pose.q[0].cpu().numpy())
        result = planner.move_to_pose_with_screw(above_cup2_target_pose)
        if not result or result == -1:
            planner.close()
            return result

    exit()  

    # Step 5 — translate from cup1 to cup2 while preserving hand orientation

    # add safe height
    cup1_pos[2] += 0.15
    cup2_pos[2] += 0.15

    delta = cup2_pos - cup1_pos   # world translation between cups
    tcp_p = tcp_pose.p[0].cpu().numpy()
    tcp_q = tcp_pose.q[0].cpu().numpy()

    # Move directly to target position in one straight line
    target_pos = tcp_p + delta
    target_pose = sapien.Pose(p=target_pos, q=tcp_q)
    result = planner.move_to_pose_with_screw(target_pose)
    if not result or result == -1:
        planner.close()
        return result


    
