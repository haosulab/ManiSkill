# Code from Lerobot koch setup tutorial:
# https://github.com/huggingface/lerobot/blob/963738d983480b1cd19295b2cb0630d0cf5c5bb5/examples/7_get_started_with_real_robot.md
import time
from dataclasses import dataclass

import gymnasium as gym
import h5py
import numpy as np
import torch
import tyro
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.utils import busy_wait
from tqdm.auto import tqdm

from mani_skill.agents.robots import Koch

# from mani_skill.utils import common, io_utils, wrappers
from mani_skill.utils import common, gym_utils, io_utils


@dataclass
class Args:
    traj_path: str
    """Path to the trajectory .h5 file to replay"""
    calibration_dir: str = ""
    """calibration directory of the robot"""
    # TODO(xhin): modify lerobot manipulator robot so we don't have to include leader arm
    leader_port: str = ""
    """port of the leader arm"""
    follower_port: str = ""
    """port of the follower arm"""
    fps: float = 30
    """frames per second for robot control"""

    # TODO(xhin): add robot option and controller conversion options for more general script
    # TODO(xhin): save video option similar to sim replay_trajectory - show greenscreen sim to compare against real world


# TODO(xhin): generalize function to other robots when support added, use yaml files like lerobot
def make_real_robot(leader_port, follower_port, calibration_dir):
    leader_arm = DynamixelMotorsBus(
        port=leader_port,
        motors={
            # name: (index, model)
            "shoulder_pan": (1, "xl330-m077"),
            "shoulder_lift": (2, "xl330-m077"),
            "elbow_flex": (3, "xl330-m077"),
            "wrist_flex": (4, "xl330-m077"),
            "wrist_roll": (5, "xl330-m077"),
            "gripper": (6, "xl330-m077"),
        },
    )

    follower_arm = DynamixelMotorsBus(
        port=follower_port,
        motors={
            # name: (index, model)
            "shoulder_pan": (1, "xl430-w250"),
            "shoulder_lift": (2, "xl430-w250"),
            "elbow_flex": (3, "xl330-m288"),
            "wrist_flex": (4, "xl330-m288"),
            "wrist_roll": (5, "xl330-m288"),
            "gripper": (6, "xl330-m288"),
        },
    )

    robot = ManipulatorRobot(
        robot_type="koch",
        leader_arms={"main": leader_arm},
        follower_arms={"main": follower_arm},
        calibration_dir=calibration_dir,
    )

    return robot


if __name__ == "__main__":
    args = args = tyro.cli(Args)
    # Load HDF5 containing trajectories
    traj_path = args.traj_path
    ori_h5_file = h5py.File(traj_path, "r")

    # Load associated json
    json_path = traj_path.replace(".h5", ".json")
    json_data = io_utils.load_json(json_path)

    env_info = json_data["env_info"]
    env_id = env_info["env_id"]
    ori_env_kwargs = env_info["env_kwargs"]

    # Create a twin env with the original kwargs
    ori_env = gym.make(env_id, **ori_env_kwargs)

    episodes = json_data["episodes"]
    n_ep = len(episodes)

    assert (
        n_ep == 1
    ), f"only one episode can replay on real robot, {traj_path} contains {n_ep} epsiodes"

    # Replay
    ep = episodes[0]
    episode_id = ep["episode_id"]
    traj_id = f"traj_{episode_id}"

    assert traj_id in ori_h5_file, tqdm.write(
        f"{traj_id} does not exist in {traj_path}"
    )

    ori_control_mode = ep["control_mode"]

    ori_actions = ori_h5_file[traj_id]["actions"][:]

    assert (
        ori_control_mode == f"pd_joint_delta_pos"
    ), "MS kochv1.1 robot should be trained with pd_joint_delta_pos controller, instead of {ori_control_mode}"

    low = torch.tensor(ori_env.agent.controller.config.lower)
    high = torch.tensor(ori_env.agent.controller.config.upper)

    ori_actions = (
        torch.tensor(ori_actions)
        if not isinstance(ori_actions, torch.Tensor)
        else ori_actions
    )
    low = torch.tensor(low) if not isinstance(low, torch.Tensor) else low
    high = torch.tensor(high) if not isinstance(high, torch.Tensor) else high

    robot = make_real_robot(args.leader_port, args.follower_port, args.calibration_dir)
    robot.connect()

    assert args.fps < 200, f"framerate of {args.fps} not in safe range"

    # conversion from delta_qpos to qpos control
    # lerobot uses arm positions in degrees - MS3 uses rads

    # first, match robot resting positon:
    # use 3 seconds to set robot to the rest position
    print("Moving to rest keyframe")
    ep_start_pos = torch.from_numpy(ori_env.agent.keyframes["rest"].qpos).float()
    for i in range(int(3 * args.fps)):
        start_loop_t = time.perf_counter()
        prev_arm_qpos = torch.deg2rad(
            torch.tensor(robot.follower_arms["main"].read("Present_Position"))
        )
        dist_to_start = ep_start_pos - prev_arm_qpos
        # we can send the dist to start directly in, since we clip the actions
        delta_qpos = gym_utils.clip_and_scale_action(dist_to_start, low, high)
        arm_action = torch.rad2deg(prev_arm_qpos + delta_qpos)
        robot.send_action(arm_action)
        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / args.fps - dt_s)

    print("Beginning trajectory")
    for action in ori_actions:
        start_loop_t = time.perf_counter()
        prev_arm_qpos = torch.deg2rad(
            torch.tensor(robot.follower_arms["main"].read("Present_Position"))
        )
        # TODO(xhin): figure out why joint -3 is backwards?
        # action[-3] *= -1
        delta_qpos = gym_utils.clip_and_scale_action(action, low, high)
        arm_action = torch.rad2deg(prev_arm_qpos + delta_qpos)
        robot.send_action(arm_action)
        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / args.fps - dt_s)
    print("trajectory ended")

    robot.disconnect()

    # Cleanup
    ori_env.close()
    ori_h5_file.close()
