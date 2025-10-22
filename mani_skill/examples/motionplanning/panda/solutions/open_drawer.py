import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks import OpenDrawerV1Env
from mani_skill.utils import common
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb
from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply

def solve(env: OpenDrawerV1Env, seed=None, debug=False, vis=False):
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
    FINGER_LENGTH = 0.025
    env = env.unwrapped
    handle_obj = env.handle_link_goal
    handle_pose_0 = handle_obj.pose 

    # Add rotational noise to the target grasp pose
    rotation_noise_range = np.deg2rad(1.0)
    rotation_noise = [np.random.uniform(-rotation_noise_range, rotation_noise_range) for _ in range(3)]
    target_rotation = euler2quat(np.pi / 2 + rotation_noise[0], -np.pi + rotation_noise[1], np.pi / 2 + rotation_noise[2]) # this was working

    # Configure target poses
    grasp_pose = sapien.Pose(q=target_rotation, p=handle_obj.pose.p.cpu().numpy()[0])
    reach_pose = grasp_pose * sapien.Pose([0, 0.0, -0.05])
    handle_target_pose = sapien.Pose(q=target_rotation, p=handle_pose_0.p.cpu().numpy()[0])
    handle_target_pose.p -= np.array([0.2, 0, 0])

    res = planner.move_to_pose_with_screw(reach_pose, dry_run=True)
    if res == -1:
        print("Failed")
        return res


    # Execute
    planner.open_gripper()
    planner.move_to_pose_with_screw(reach_pose)
    planner.open_gripper()
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()
    planner.move_to_pose_with_screw(handle_target_pose)
    res = planner.open_gripper()
    planner.close()
    return res
