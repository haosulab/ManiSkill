"""
Code based on https://github.com/huggingface/lerobot for supporting real robot control via the unified LeRobot interface.
"""

import time
from typing import List, Optional

import numpy as np
import torch

from mani_skill.agents.base_real_agent import BaseRealAgent
from mani_skill.utils import common
from mani_skill.utils.structs.types import Array

try:
    from lerobot.common.robot_devices.cameras.utils import Camera
    from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
    from lerobot.common.robot_devices.utils import busy_wait
except ImportError:
    pass


class LeRobotRealAgent(BaseRealAgent):
    """
    LeRobotRealAgent is a general class for controlling real robots via the LeRobot system. You simply just pass in the ManipulatorRobot instance you create via LeRobot and pass it here to make it work with ManiSkill Sim2Real environment interfaces.

    Args:
        robot (ManipulatorRobot): The ManipulatorRobot instance you create via LeRobot.
        use_cached_qpos (bool): Whether to cache the fetched qpos values. If True, the qpos will be
            read from the cache instead of the real robot when possible. This cache is only invalidated when
            set_target_qpos or set_target_qvel is called. This can be useful if you want to easily have higher frequency (> 30Hz) control since qpos reading from the robot is
            currently the slowest part of LeRobot for some of the supported motors.
    """

    def __init__(self, robot: ManipulatorRobot, use_cached_qpos: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._captured_sensor_data = None
        self.real_robot = robot
        self.use_cached_qpos = use_cached_qpos
        self._cached_qpos = None

    def start(self):
        self.real_robot.connect()

    def stop(self):
        self.real_robot.disconnect()

    def set_target_qpos(self, qpos: Array):
        self._cached_qpos = None
        qpos = common.to_cpu_tensor(qpos).flatten()
        self.real_robot.send_action(torch.rad2deg(qpos))

    def reset(self, qpos: Array):
        qpos = common.to_cpu_tensor(qpos)
        freq = 60
        target_pos = self.qpos
        max_rad_per_step = 0.01
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
        qpos_deg = self.real_robot.follower_arms["main"].read("Present_Position")
        qpos = torch.deg2rad(torch.tensor(qpos_deg)).unsqueeze(0)
        self._cached_qpos = qpos
        return qpos

    def get_qvel(self):
        raise NotImplementedError
