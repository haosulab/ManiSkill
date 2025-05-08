import argparse
import signal

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

signal.signal(signal.SIGINT, signal.SIG_DFL)  # allow ctrl+c
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, visualization
from mani_skill.utils.wrappers import RecordEpisode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, required=True)
    parser.add_argument("-o", "--obs-mode", type=str)
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-c", "--control-mode", type=str, default="pd_ee_delta_pose")
    parser.add_argument("--render-mode", type=str, default="sensors")
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
        render_mode=args.render_mode,
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

    obs, _ = env.reset()
    after_reset = True

    # Viewer
    if args.enable_sapien_viewer:
        env.render_human()
    renderer = visualization.ImageRenderer(wait_for_button_press=False)
    # disable all default plt shortcuts that are lowercase letters
    plt.rcParams["keymap.fullscreen"].remove("f")
    plt.rcParams["keymap.home"].remove("h")
    plt.rcParams["keymap.home"].remove("r")
    plt.rcParams["keymap.back"].remove("c")
    plt.rcParams["keymap.forward"].remove("v")
    plt.rcParams["keymap.pan"].remove("p")
    plt.rcParams["keymap.zoom"].remove("o")
    plt.rcParams["keymap.save"].remove("s")
    plt.rcParams["keymap.grid"].remove("g")
    plt.rcParams["keymap.yscale"].remove("l")
    plt.rcParams["keymap.xscale"].remove("k")

    def render_wait():
        if not args.enable_sapien_viewer:
            return
        while True:
            env.render_human()
            sapien_viewer = env.viewer
            if sapien_viewer.window.key_down("0"):
                break

    # Embodiment
    has_base = "base" in env.agent.controller.configs
    num_arms = sum("arm" in x for x in env.agent.controller.configs)
    has_gripper = any("gripper" in x for x in env.agent.controller.configs)
    gripper_action = 1
    EE_ACTION = 0.1

    # For real-time simulation
    import time

    control_timestep = env.unwrapped.control_timestep
    last_update_time = time.time()

    while True:
        current_time = time.time()
        elapsed_time = current_time - last_update_time

        if elapsed_time >= control_timestep:
            last_update_time = current_time

            # -------------------------------------------------------------------------- #
            # Visualization
            # -------------------------------------------------------------------------- #
            if args.enable_sapien_viewer:
                env.render_human()

            render_frame = env.render().cpu().numpy()[0]

            if after_reset:
                after_reset = False
                if args.enable_sapien_viewer:
                    renderer.close()
                    renderer = visualization.ImageRenderer(wait_for_button_press=False)

            renderer(render_frame)

            # -------------------------------------------------------------------------- #
            # Interaction
            # -------------------------------------------------------------------------- #
            # Get the set of currently pressed keys
            pressed_keys = renderer.pressed_keys

            body_action = np.zeros([3])
            base_action = np.zeros([2])  # hardcoded for fetch robot

            # Parse end-effector action
            if (
                "pd_ee_delta_pose" in args.control_mode
                or "pd_ee_target_delta_pose" in args.control_mode
            ):
                ee_action = np.zeros([6])
            elif (
                "pd_ee_delta_pos" in args.control_mode
                or "pd_ee_target_delta_pos" in args.control_mode
            ):
                ee_action = np.zeros([3])
            else:
                raise NotImplementedError(args.control_mode)

            if has_base:
                if "w" in pressed_keys:  # forward
                    base_action[0] = 1
                if "s" in pressed_keys:  # backward
                    base_action[0] = -1
                if "a" in pressed_keys:  # rotate counter
                    base_action[1] = 1
                if "d" in pressed_keys:  # rotate clockwise
                    base_action[1] = -1
                if "z" in pressed_keys:  # lift
                    body_action[2] = 1
                if "x" in pressed_keys:  # lower
                    body_action[2] = -1
                if "v" in pressed_keys:  # rotate head left
                    body_action[0] = 1
                if "b" in pressed_keys:  # rotate head right
                    body_action[0] = -1
                if "n" in pressed_keys:  # tilt head down
                    body_action[1] = 1
                if "m" in pressed_keys:  # rotate head up
                    body_action[1] = -1

            # End-effector
            if num_arms > 0:
                # Position
                if "i" in pressed_keys:  # +x
                    ee_action[0] = EE_ACTION
                if "k" in pressed_keys:  # -x
                    ee_action[0] = -EE_ACTION
                if "j" in pressed_keys:  # +y
                    ee_action[1] = EE_ACTION
                if "l" in pressed_keys:  # -y
                    ee_action[1] = -EE_ACTION
                if "u" in pressed_keys:  # +z
                    ee_action[2] = EE_ACTION
                if "o" in pressed_keys:  # -z
                    ee_action[2] = -EE_ACTION

                # Rotation (axis-angle)
                if "1" in pressed_keys:
                    ee_action[3:6] = (1, 0, 0)
                elif "2" in pressed_keys:
                    ee_action[3:6] = (-1, 0, 0)
                elif "3" in pressed_keys:
                    ee_action[3:6] = (0, 1, 0)
                elif "4" in pressed_keys:
                    ee_action[3:6] = (0, -1, 0)
                elif "5" in pressed_keys:
                    ee_action[3:6] = (0, 0, 1)
                elif "6" in pressed_keys:
                    ee_action[3:6] = (0, 0, -1)

            # Gripper
            if has_gripper:
                if "f" in pressed_keys:  # open gripper
                    gripper_action = 1
                if "g" in pressed_keys:  # close gripper
                    gripper_action = -1

            # Other functions
            if "0" in pressed_keys:  # switch to SAPIEN viewer
                render_wait()
                if "0" in renderer.pressed_keys:
                    renderer.pressed_keys.remove("0")
            elif "r" in pressed_keys:  # reset env
                obs, _ = env.reset()
                gripper_action = 1
                after_reset = True
                if "r" in renderer.pressed_keys:
                    renderer.pressed_keys.remove("r")
                continue
            elif "q" in pressed_keys or "escape" in pressed_keys:  # exit
                break

            # Visualize observation
            if "v" in pressed_keys and "pointcloud" in env.obs_mode:
                import trimesh

                xyzw = obs["pointcloud"]["xyzw"]
                mask = xyzw[..., 3] > 0
                rgb = obs["pointcloud"]["rgb"]
                if "robot_seg" in obs["pointcloud"]:
                    robot_seg = obs["pointcloud"]["robot_seg"]
                    rgb = np.uint8(robot_seg * [11, 61, 127])
                trimesh.PointCloud(xyzw[mask, :3], rgb[mask]).show()
                if "v" in renderer.pressed_keys:
                    renderer.pressed_keys.remove("v")

            # -------------------------------------------------------------------------- #
            # Post-process action
            # -------------------------------------------------------------------------- #
            action_dict = dict(
                base=base_action,
                arm=ee_action,
                body=body_action,
                gripper=gripper_action,
            )
            action_dict = common.to_tensor(action_dict)
            action = env.agent.controller.from_action_dict(action_dict)

            obs, reward, terminated, truncated, info = env.step(action)
            print("reward", reward)
            print("terminated", terminated, "truncated", truncated)
            print("info", info)
        else:
            # Small sleep to prevent CPU hogging when waiting for the next timestep
            time.sleep(0.001)
            # Update the renderer to keep the display responsive
            plt.pause(0.001)

    env.close()


if __name__ == "__main__":
    main()
