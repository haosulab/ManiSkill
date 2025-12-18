import numpy as np
import sapien

from mani_skill.envs.tasks.tabletop.place_dish_in_rack import PlaceDishInRackEnv
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)


def solve(env: PlaceDishInRackEnv, seed=None, debug=False, vis=False):
    """Grasp flat plate from the rim and lift it up."""
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
    )

    env_sim = env.unwrapped

    # Get plate position and orientation
    plate_pose = env_sim.plate.pose
    plate_pos = plate_pose.p[0].cpu().numpy()

    if debug:
        print(f"\n=== PLATE POSITION ===")
        print(f"Plate position: {plate_pos}")

    # Plate is flat on table (outer radius=0.085m, rim height=0.018m)
    plate_outer_radius = env_sim._plate_outer_radius
    plate_inner_radius = env_sim._plate_inner_radius
    plate_rim_height = env_sim._plate_rim_height
    plate_base_thickness = env_sim._plate_base_thickness

    # Approach from top-down to grasp the rim
    approaching = np.array([0, 0, -1])  # Approach from above

    # Closing direction horizontal to pinch opposite sides of the rim
    closing = np.array([1, 0, 0])

    # Position the grasp point on the RIGHT SIDE of the plate
    # Approach from top-down, gripper fingers will pinch left-right across the rim
    center = plate_pos.copy()

    # Offset to the RIGHT side (positive X direction) - position over the rim
    # rim_grasp_radius = (plate_outer_radius + plate_inner_radius) / 2.0
    rim_grasp_radius = 1.25*plate_outer_radius # ( + plate_inner_radius) / 2.0
    
    center[0] = plate_pos[0] + rim_grasp_radius - 0.03 # Move to right side in X direction

    # Height should be lower on the rim for a more secure grip
    # center[2] = plate_pos[2] + plate_base_thickness + plate_rim_height * 0.3  # Low-mid on rim
    center[2] = plate_pos[2] + plate_base_thickness + plate_rim_height/2 - 0.01  # Low-mid on rim


    # Build grasp pose
    grasp_pose = env_sim.agent.build_grasp_pose(approaching, closing, center)

    if debug:
        print(f"\n=== GRASP INFO ===")
        print(f"Plate center: {plate_pos}")
        print(f"Grasp center (at rim): {center}")
        print(f"Plate outer radius: {plate_outer_radius}")
        print(f"Approaching: {approaching}")
        print(f"Closing: {closing}")

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 1: REACH ===")

    # Back away 8cm before approaching (move opposite to approaching direction in local frame)
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.1])
    result = planner.move_to_pose_with_RRTConnect(reach_pose)
    if result == -1:
        if debug:
            print("❌ Failed to reach")
        planner.close()
        return result

    if debug:
        print("✓ Reached approach position")

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 2: GRASP ===")

    result = planner.move_to_pose_with_RRTConnect(grasp_pose)
    if result == -1:
        if debug:
            print("❌ Failed to grasp position")
        planner.close()
        return result

    # Close gripper with maximum force and longer duration to prevent slipping
    planner.close_gripper(t=40, gripper_state=-1.0)  # Close for 40 steps with full force

    # Let physics settle after grasping to ensure firm grip
    qpos = env_sim.agent.robot.get_qpos()[0, : len(planner.planner.joint_vel_limits)].cpu().numpy()
    for i in range(20):  # Hold position for 20 more steps
        if planner.control_mode == "pd_joint_pos":
            action = np.hstack([qpos, -1.0])
        else:
            action = np.hstack([qpos, qpos * 0, -1.0])
        env_sim.step(action)

    is_grasped = env_sim.agent.is_grasping(env_sim.plate)[0].item()

    if debug:
        gripper_qpos = env_sim.agent.robot.get_qpos()[0, 7:9].cpu().numpy()
        print(f"Gripper closed: {gripper_qpos}")
        print(f"Is grasped: {is_grasped}")

    # -------------------------------------------------------------------------- #
    # Lift
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 3: LIFT ===")

    lift_pose = sapien.Pose([0, 0, 0.15]) * grasp_pose
    res = planner.move_to_pose_with_screw(lift_pose)

    if debug:
        plate_after_lift = env_sim.plate.pose.p[0].cpu().numpy()
        print(f"✓ Lifted plate to: {plate_after_lift}")

    # -------------------------------------------------------------------------- #
    # Move to rack FIRST (keep plate horizontal for stability during transport)
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 4: MOVE TO RACK (HORIZONTAL) ===")

    # Get rack position and dimensions
    rack_pos = env_sim.dish_rack.pose.p[0].cpu().numpy()
    rack_height = env_sim._rack_extent[2]  # 0.085m
    rack_depth = env_sim._rack_extent[1]  # 0.168m

    # Position directly above rack CENTER
    target_pos = rack_pos.copy()
    # No Y offset - keep at rack center Y
    target_pos[0] += 0.05  # Centered in X
    target_pos[1] += 0.03  # 10cm above rack for clearance
    target_pos[2] += rack_height + 0.10  # 10cm above rack for clearance

    # Keep horizontal orientation (same as lift pose)
    transport_pose = sapien.Pose(p=target_pos, q=lift_pose.q)

    res = planner.move_to_pose_with_RRTConnect(transport_pose)
    if res == -1:
        if debug:
            print("❌ Failed to move to rack")
        planner.close()
        return res

    if debug:
        plate_at_rack = env_sim.plate.pose.p[0].cpu().numpy()
        print(f"✓ Moved to rack center (horizontal): {target_pos}")
        print(f"  Plate is at: {plate_at_rack}")

    # -------------------------------------------------------------------------- #
    # Rotate to vertical above rack center
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 5: ROTATE TO VERTICAL ABOVE RACK ===")

    # Rotate 90 degrees to make plate vertical
    rotation = sapien.Pose(q=[0.7071068, 0, 0.7071068, 0])  # 90 deg around Y
    vertical_pose = transport_pose * rotation

    res = planner.move_to_pose_with_screw(vertical_pose)

    if debug:
        plate_vertical = env_sim.plate.pose.p[0].cpu().numpy()
        print(f"✓ Rotated to vertical: {plate_vertical}")


    # -------------------------------------------------------------------------- #
    # Release plate
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 6: RELEASE ===")

    release_pose = vertical_pose * sapien.Pose([-0.05, 0, 0])  # Move down 5cm to release position
    res = planner.move_to_pose_with_RRTConnect(release_pose)
    planner.open_gripper()

    if debug:
        final_plate_pos = env_sim.plate.pose.p[0].cpu().numpy()
        print(f"✓ Released plate at: {final_plate_pos}")

    # -------------------------------------------------------------------------- #
    # Move back to safe position
    # -------------------------------------------------------------------------- #
    if debug:
        print(f"\n=== STEP 7: RETURN TO SAFE POSITION ===")

    # Move up and back to a safe position away from the plate
    retreat_pose = vertical_pose * sapien.Pose([0, 0, -0.15])  # Move up 15cm from release position
    res = planner.move_to_pose_with_RRTConnect(retreat_pose)

    if debug:
        if res == -1:
            print("❌ Failed to retreat")
        else:
            print("✓ Moved to safe retreat position")
        print(f"✓ Task complete!")

    planner.close()
    return res
