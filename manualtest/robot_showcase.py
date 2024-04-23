import sys

import gymnasium as gym

import mani_skill.envs
from mani_skill.envs.sapien_env import BaseEnv

if __name__ == "__main__":
    robot = sys.argv[1]
    # robot in ["panda", "fetch", "xmate3_robotiq"]:
    env = gym.make(
        "Empty-v1",
        enable_shadow=True,
        robot_uids=robot,
        render_mode="human",
        control_mode="pd_joint_pos"
        # control_mode="arm_pd_ee_delta_pose_gripper_pd_joint_pos",
        # shader_dir="rt-fast",
    )
    env.reset(seed=0)
    env: BaseEnv = env.unwrapped
    keyframe = env.agent.keyframes["standing"]
    env.agent.robot.set_pose(keyframe.pose)
    env.agent.robot.set_qpos(keyframe.qpos)
    viewer = env.render()
    viewer.paused = True
    viewer = env.render()

    while True:
        env.step(keyframe.qpos)
        viewer = env.render()
        # if viewer.window.key_press("n"):
        #     env.close()
        #     del env
        #     break
