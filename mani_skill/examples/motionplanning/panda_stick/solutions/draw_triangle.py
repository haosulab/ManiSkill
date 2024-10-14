import numpy as np
import sapien
from mani_skill.envs.tasks import PushCubeEnv
from mani_skill.examples.motionplanning.panda_stick.motionplanner import \
    PandaStickMotionPlanningSolver
from PIL import Image


def solve(env: PushCubeEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaStickMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    rot = list(env.agent.tcp.pose.get_q()[0].cpu().numpy())
    reach_pose = sapien.Pose(p=list(env.vertices[0, 0].numpy()), q=rot)
    res = planner.move_to_pose_with_screw(reach_pose)
    # -------------------------------------------------------------------------- #
    # Move to second vertex
    # -------------------------------------------------------------------------- #

    reach_pose = sapien.Pose(p=list(env.vertices[0, 1]), q=rot)
    res = planner.move_to_pose_with_screw(reach_pose)

    reach_pose = sapien.Pose(p=list(env.vertices[0, 2]), q=rot)
    res = planner.move_to_pose_with_screw(reach_pose)

    reach_pose = sapien.Pose(p=list(env.vertices[0, 0]), q=rot)
    res = planner.move_to_pose_with_screw(reach_pose)

    planner.close()
    return res
