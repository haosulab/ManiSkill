import argparse

import gym
import numpy as np

from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.visualization.cv2_utils import OpenCVViewer
from mani_skill2.utils.wrappers import RecordEpisode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, required=True)
    parser.add_argument("-o", "--obs-mode", type=str)
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-c", "--control-mode", type=str, default="pd_ee_delta_pose")
    parser.add_argument("--render-mode", type=str, default="cameras")
    parser.add_argument("--enable-sapien-viewer", action="store_true")
    parser.add_argument("--record-dir", type=str)
    args, opts = parser.parse_known_args()

    # Parse env kwargs
    print("opts:", opts)
    eval_str = lambda x: eval(x[1:]) if x.startswith("@") else x
    env_kwargs = dict((x, eval_str(y)) for x, y in zip(opts[0::2], opts[1::2]))
    print("env_kwargs:", env_kwargs)
    args.env_kwargs = env_kwargs

    return args


def main():
    np.set_printoptions(suppress=True, precision=3)
    args = parse_args()

    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        **args.env_kwargs
    )

    record_dir = args.record_dir
    if record_dir:
        record_dir = record_dir.format(env_id=args.env_id)
        env = RecordEpisode(env, record_dir, render_mode=args.render_mode)

    print("Observation space", env.observation_space)
    print("Action space", env.action_space)
    print("Control mode", env.control_mode)
    print("Reward mode", env.reward_mode)

    obs = env.reset()
    after_reset = True

    # Viewer
    if args.enable_sapien_viewer:
        env.render(mode="human")
    opencv_viewer = OpenCVViewer()

    def render_wait():
        if not args.enable_sapien_viewer:
            return
        while True:
            sapien_viewer = env.render(mode="human")
            if sapien_viewer.window.key_down("0"):
                break

    has_gripper = "gripper" in env.agent.controller.configs
    gripper_action = 1
    EE_ACTION = 0.1

    while True:
        # -------------------------------------------------------------------------- #
        # Visualization
        # -------------------------------------------------------------------------- #
        if args.enable_sapien_viewer:
            env.render(mode="human")

        render_frame = env.render(mode=args.render_mode)

        if after_reset:
            after_reset = False
            # Re-focus on opencv viewer
            if args.enable_sapien_viewer:
                opencv_viewer.close()
                opencv_viewer = OpenCVViewer()

        # -------------------------------------------------------------------------- #
        # Interaction
        # -------------------------------------------------------------------------- #
        # Input
        key = opencv_viewer.imshow(render_frame)

        # Parse end-effector action
        if args.control_mode in ["pd_ee_delta_pose", "pd_ee_target_delta_pose"]:
            ee_action = np.zeros([6])
        elif args.control_mode in ["pd_ee_delta_pos", "pd_ee_target_delta_pos"]:
            ee_action = np.zeros([3])
        else:
            raise NotImplementedError(args.control_mode)

        # Position
        if key == "i":  # +x
            ee_action[0] = EE_ACTION
        elif key == "k":  # -x
            ee_action[0] = -EE_ACTION
        elif key == "j":  # +y
            ee_action[1] = EE_ACTION
        elif key == "l":  # -y
            ee_action[1] = -EE_ACTION
        elif key == "u":  # +z
            ee_action[2] = EE_ACTION
        elif key == "o":  # -z
            ee_action[2] = -EE_ACTION

        # Rotation (axis-angle)
        if key == "1":
            ee_action[3:6] = (1, 0, 0)
        elif key == "2":
            ee_action[3:6] = (-1, 0, 0)
        elif key == "3":
            ee_action[3:6] = (0, 1, 0)
        elif key == "4":
            ee_action[3:6] = (0, -1, 0)
        elif key == "5":
            ee_action[3:6] = (0, 0, 1)
        elif key == "6":
            ee_action[3:6] = (0, 0, -1)

        # Parse
        if key == "f":  # open gripper
            gripper_action = 1
        elif key == "g":  # close gripper
            gripper_action = -1

        # Other functions
        if key == "0":  # switch to SAPIEN viewer
            render_wait()
        elif key == "r":  # reset env
            obs = env.reset()
            gripper_action = 1
            after_reset = True
            continue
        elif key == "q":  # exit
            break

        # Visualize observation
        if key == "v":
            if "rgbd" in env.obs_mode:
                from itertools import chain
                from mani_skill2.utils.visualization.misc import (
                    observations_to_images,
                    tile_images,
                )

                images = list(
                    chain(*[observations_to_images(x) for x in obs["image"].values()])
                )
                render_frame = tile_images(images)
                opencv_viewer.imshow(render_frame)
            elif "pointcloud" in env.obs_mode:
                import trimesh

                xyz = obs["pointcloud"]["xyzw"][..., :3]
                rgb = obs["pointcloud"]["rgb"]
                # rgb = np.tile(obs["pointcloud"]["robot_seg"] * 255, [1, 3])
                trimesh.PointCloud(xyz, rgb).show()

        action = ee_action
        if has_gripper:
            action = np.hstack([ee_action, gripper_action])

        obs, reward, done, info = env.step(action)
        print("reward", reward)
        print("done", done)
        print("info", info)

    env.close()


if __name__ == "__main__":
    main()
