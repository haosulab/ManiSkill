"""
Code based on https://github.com/huggingface/lerobot for supporting real robot control via the unified LeRobot interface.
"""

import time
from typing import List, Optional

import numpy as np
import torch

from mani_skill.agents.base_real_agent import BaseRealAgent
from mani_skill.utils import common
from mani_skill.utils.logging_utils import logging
from mani_skill.utils.structs.types import Array

try:
    from lerobot.cameras.camera import Camera
    from lerobot.motors.motors_bus import MotorNormMode
    from lerobot.robots.robot import Robot
    from lerobot.utils.robot_utils import busy_wait
except ImportError:
    pass


class LeRobotRealAgent(BaseRealAgent):
    """
    LeRobotRealAgent is a general class for controlling real robots via the LeRobot system. You simply just pass in the Robot instance you create via LeRobot and pass it here to make it work with ManiSkill Sim2Real environment interfaces.

    Args:
        robot (Robot): The Robot instance you create via LeRobot.
        use_cached_qpos (bool): Whether to cache the fetched qpos values. If True, the qpos will be
            read from the cache instead of the real robot when possible. This cache is only invalidated when
            set_target_qpos or set_target_qvel is called. This can be useful if you want to easily have higher frequency (> 30Hz) control since qpos reading from the robot is currently the slowest part of LeRobot for some of the supported motors.
        calibration_offset (dict): A dictionary of motor names mapping to their calibration offsets in degrees. This is strongly recommended to be tuned since the zero joint position of the robot motors after calibration with LeRobot does not always align with simulation. An example for the SO100 robot arm is shown below. To tune these values you should move the robot to a rest position with joint angles equal to 0 or 90 degrees, and tune the values below until the real robot also matches the simulation rendered configuration (in which all adjacent parts are at right angles / parallel to each other).

        .. code-block:: python
        calibration_offset = {
            "shoulder_pan": -3,
            "shoulder_lift": -3,
            "elbow_flex": 5,
            "wrist_flex": 5,
            "wrist_roll": 0,
            "gripper": 0,
        }
    """

    def __init__(
        self,
        robot: Robot,
        use_cached_qpos: bool = True,
        calibration_offset: dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._captured_sensor_data = None
        self.real_robot = robot
        self.use_cached_qpos = use_cached_qpos
        self._cached_qpos = None
        self._motor_keys: List[str] = None
        self._calibration_offset = calibration_offset
        if self._calibration_offset is None:
            logging.warning(
                "The calibration offset for the robot is not tuned!! Unless you are absolutely sure you don't need it,you will most likely get poor results for sim2real/real2sim transfer."
            )

        if self.real_robot.name == "so100_follower":
            self.real_robot.bus.motors["gripper"].norm_mode = MotorNormMode.DEGREES

    def start(self):
        self.real_robot.connect()

    def stop(self):
        self.real_robot.disconnect()

    def set_target_qpos(self, qpos: Array):
        self._cached_qpos = None
        qpos = common.to_cpu_tensor(qpos).flatten()
        qpos = torch.rad2deg(qpos)
        action = dict()
        for motor_key, qpos_val in zip(self._motor_keys, qpos):
            action_key = motor_key + ".pos"
            action[action_key] = qpos_val
            if motor_key in self._calibration_offset:
                action[action_key] = (
                    action[action_key] + self._calibration_offset[motor_key]
                )
        self.real_robot.send_action(action)

    def reset(self, qpos: Array):
        qpos = common.to_cpu_tensor(qpos)
        freq = 30
        target_pos = self.qpos
        max_rad_per_step = 0.025
        for _ in range(int(20 * freq)):
            start_loop_t = time.perf_counter()
            delta_step = (qpos - target_pos).clip(
                min=-max_rad_per_step, max=max_rad_per_step
            )
            if np.linalg.norm(delta_step) <= 1e-4:
                break
            target_pos += delta_step

            self.set_target_qpos(target_pos)
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / freq - dt_s)

    def capture_sensor_data(self, sensor_names: Optional[List[str]] = None):
        sensor_obs = dict()
        cameras: dict[str, Camera] = self.real_robot.cameras
        if sensor_names is None:
            sensor_names = list(cameras.keys())
        for name in sensor_names:
            data = cameras[name].async_read()
            # until https://github.com/huggingface/lerobot/issues/860 is resolved we temporarily assume this is RGB data only otherwise need to write a few extra if statements to check
            # if isinstance(cameras[name], IntelRealSenseCamera):
            sensor_obs[name] = dict(rgb=(common.to_tensor(data)).unsqueeze(0))
        self._captured_sensor_data = sensor_obs

    def get_sensor_data(self, sensor_names: Optional[List[str]] = None):
        if self._captured_sensor_data is None:
            raise RuntimeError(
                "No sensor data captured yet. Please call capture_sensor_data() first."
            )
        if sensor_names is None:
            return self._captured_sensor_data
        else:
            return {
                k: v for k, v in self._captured_sensor_data.items() if k in sensor_names
            }

    def get_qpos(self):
        # NOTE (stao): the slowest part of inference is reading the qpos from the robot. Each time it takes about 5-6 milliseconds, meaning control frequency is capped at 200Hz.
        # and if you factor in other operations like policy inference etc. the max control frequency is typically more like 30-60 Hz.
        # Moreover on the rare occassions reading qpos can take 40 milliseconds which causes the control step to fall behind the desired control frequency.
        if self.use_cached_qpos and self._cached_qpos is not None:
            return self._cached_qpos.clone()
        qpos_deg = self.real_robot.bus.sync_read("Present_Position")

        # NOTE (stao): It seems the calibration from LeRobot has some offsets in some joints. We fix reading them here to match the expected behavior
        if self._calibration_offset is not None:
            for k in self._calibration_offset.keys():
                qpos_deg[k] = qpos_deg[k] - self._calibration_offset[k]
        if self._motor_keys is None:
            self._motor_keys = list(qpos_deg.keys())
        qpos_deg = common.flatten_state_dict(qpos_deg)
        qpos = torch.deg2rad(torch.tensor(qpos_deg)).unsqueeze(0)
        self._cached_qpos = qpos
        return qpos

    def get_qvel(self):
        raise NotImplementedError
