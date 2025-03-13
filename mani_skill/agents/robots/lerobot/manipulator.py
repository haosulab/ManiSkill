"""
Code based on https://github.com/huggingface/lerobot for supporting real robot control via the unified LeRobot interface.
"""

import time

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


class LeRobotAgent(BaseRealAgent):
    """
    LeRobotAgent is a class for controlling a real robot. You simply just pass in the ManipulatorRobot instance you create via LeRobot and pass it here to make it work with ManiSkill real environment interfaces.
    """

    def __init__(self, robot: ManipulatorRobot, **kwargs):
        super().__init__(**kwargs)
        self._captured_sensor_data = None
        self.robot = robot

    def start(self):
        self.robot.connect()

    def stop(self):
        self.robot.disconnect()

    def set_target_qpos(self, qpos: Array):
        qpos = common.to_cpu_tensor(qpos)
        self.robot.send_action(torch.rad2deg(qpos))

    def set_target_qvel(self, qvel: Array):
        qvel = common.to_cpu_tensor(qvel)
        self.robot.send_action(qvel)

    def reset(self, qpos: Array):
        qpos = common.to_cpu_tensor(qpos)
        freq = 60
        target_pos = self.qpos
        max_rad_per_step = 0.01
        for _ in range(int(5 * freq)):
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

    def capture_sensor_data(self):
        sensor_obs = dict()
        cameras: dict[str, Camera] = self.robot.cameras
        for name in cameras:
            data = cameras[name].async_read()
            # until https://github.com/huggingface/lerobot/issues/860 is resolved we temporarily assume this is RGB data only otherwise need to write a few extra if statements to check
            # if isinstance(cameras[name], IntelRealSenseCamera):
            sensor_obs[name] = dict(rgb=data)
        self._captured_sensor_data = sensor_obs

    def get_sensor_obs(self):
        if self._captured_sensor_data is None:
            raise RuntimeError(
                "No sensor data captured yet. Please call capture_sensor_data() first."
            )
        return self._captured_sensor_data

    def get_qpos(self):
        return torch.deg2rad(
            torch.tensor(self.robot.follower_arms["main"].read("Present_Position"))
        )

    def get_qvel(self):
        return None
