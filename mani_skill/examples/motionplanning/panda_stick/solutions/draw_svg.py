
import sapien

from mani_skill.examples.motionplanning.panda_stick.motionplanner import \
    PandaStickMotionPlanningSolver
from PIL import Image


def solve(env, seed=None, debug=False, vis=False):
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
    env.reset()
    

    rot = list(env.agent.tcp.pose.get_q()[0].cpu().numpy())
    res = None
    for point in env.points[0]:
        reach_pose = sapien.Pose(p=list(point.cpu().numpy()), q=rot)
        res = planner.move_to_pose_with_screw(reach_pose)

    planner.close()
    return res