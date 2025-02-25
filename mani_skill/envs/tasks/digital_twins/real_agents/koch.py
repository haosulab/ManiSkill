import time
from typing import Optional

import cv2
import hydra
import numpy as np
import torch

# lerobot/lerobot-related imports
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.motors.configs import DynamixelMotorsBusConfig
from lerobot.common.robot_devices.robots.configs import KochRobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.utils import busy_wait

# lerobot/lerobot-related imports
from tqdm import tqdm

from mani_skill.agents.robots.koch.koch import Koch
from mani_skill.envs.tasks.digital_twins.base_real_agent import BaseRealAgent
from mani_skill.utils import common
from mani_skill.utils.visualization.misc import tile_images


class MS3RealKoch(BaseRealAgent):
    def __init__(
        self,
        control_freq: int,
        control_mode: Optional[str] = None,
        img_square_crop=True,
        img_res=(128, 128),
        **kwargs
    ):
        super().__init__(
            sim_agent_cls=Koch,
            control_freq=control_freq,
            control_mode=control_mode,
            **kwargs,
        )
        self.img_square_crop = img_square_crop
        self.img_res = img_res

    def _load_agent(self, **kwargs):
        """Conect the robot"""
        robot_config = KochRobotConfig(
            leader_arms={},
            follower_arms={
                "main": DynamixelMotorsBusConfig(
                    port="/dev/ttyACM0",  # <--- CHANGE HERE
                    motors={
                        # name: (index, model)
                        "shoulder_pan": [1, "xl430-w250"],
                        "shoulder_lift": [2, "xl430-w250"],
                        "elbow_flex": [3, "xl330-m288"],
                        "wrist_flex": [4, "xl330-m288"],
                        "wrist_roll": [5, "xl330-m288"],
                        "gripper": [6, "xl330-m288"],
                    },
                ),
            },
            cameras={
                "base_camera": OpenCVCameraConfig(
                    camera_index=2,  # <--- CHANGE HERE
                    fps=60,
                    width=640,
                    height=480,
                    rotation=90,  # <--- CHANGE If Necessary
                ),
            },
            calibration_dir="koch_calibration",  # <--- CHANGE HERE
        )
        robot = ManipulatorRobot(robot_config)
        robot.connect()
        return robot

    def send_qpos(self, qpos):
        """send qpos in radians for robot to match"""
        self.robot.send_action(torch.rad2deg(qpos))

    @property
    def qpos(self):
        """Read current qpos in radians of robot"""
        return torch.deg2rad(
            torch.tensor(self.robot.follower_arms["main"].read("Present_Position"))
        )

    @property
    def qvel(self):
        """Read current qvel of robot"""
        raise NotImplementedError()

    # TODO (xhin): write more concise reset
    def reset(self, qpos: torch.Tensor = None):
        """Returns robot to given qpos"""
        freq = 60
        max_rad_per_step = 0.01
        target_pos = self.qpos  # base_target, to be updated during iteration
        if qpos is not None:
            assert torch.all(qpos <= np.pi) and torch.all(
                qpos >= -np.pi
            ), "qpos expected in range [-np.pi, np.pi]"
            print(
                "Moving to initial keyframe, ensure robot is in rest position and press Enter"
            )
            input()
            for _ in tqdm(range(int(5 * freq))):
                start_loop_t = time.perf_counter()
                delta_step = (qpos - target_pos).clip(
                    min=-max_rad_per_step, max=max_rad_per_step
                )
                if np.linalg.norm(delta_step) <= 1e-4:
                    print("converged to init pose")
                    break
                target_pos += delta_step
                self.send_qpos(target_pos)
                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / freq - dt_s)

            print("Press Enter to Reset Koch v1.1 Follower Arm")
            input()

    # TODO (xhin): test multi-camera setup
    def render(self):
        images = []
        for name in self.robot.cameras:
            images.append(torch.tensor(self.img_trans(self.robot.cameras[name].read())))
        return tile_images(images)

    # TODO (xhin): test multi-camera setup
    def get_obs_sensor_data(self):
        sensor_obs = dict()
        for name in self.robot.cameras:
            img = torch.tensor(self.img_trans(self.robot.cameras[name].read()))
            sensor_obs[name] = dict(rgb=img)
        return common.batch(sensor_obs)

    def img_trans(self, img):
        # center crop
        if self.img_square_crop:
            xy_res = img.shape[:2]
            crop_res = np.min(xy_res)
            cutoff = (np.max(xy_res) - crop_res) // 2
            if xy_res[0] == xy_res[1]:
                pass
            elif np.argmax(xy_res) == 0:
                img = img[cutoff:-cutoff, :, :]
            else:
                img = img[:, cutoff:-cutoff, :]
        # resize
        img = cv2.resize(img, self.img_res)
        return img
