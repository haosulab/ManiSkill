import time
from typing import Optional

import hydra
import numpy as np
import torch
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
from lerobot.common.robot_devices.utils import busy_wait

# lerobot/lerobot-related imports
from lerobot.common.utils.utils import init_hydra_config
from tqdm import tqdm

from mani_skill.agents.robots.koch.koch import Koch
from mani_skill.envs.tasks.digital_twins.base_real_agent import BaseRealAgent
from mani_skill.utils import common
from mani_skill.utils.visualization.misc import tile_images


class MS3RealKoch(BaseRealAgent):
    def __init__(
        self, yaml_path, control_freq: int, control_mode: Optional[str] = None, **kwargs
    ):
        self.yaml_path = yaml_path
        # default image transformation options, feel free to change if you set up your camera(s) differently
        kwargs.setdefault("img_square_crop", True)
        kwargs.setdefault("img_rotate", True)
        kwargs.setdefault("img_res", (128, 128))
        super().__init__(
            sim_agent_cls=Koch,
            control_freq=control_freq,
            control_mode=control_mode,
            **kwargs,
        )

    def _load_agent(self, **kwargs):
        """Conect the robot"""
        # load koch with provided yaml
        robot = hydra.utils.instantiate(init_hydra_config(self.yaml_path))
        robot.connect()

        # turn off leader arm torque - unnecessary for pure rl
        leader = list(robot.leader_arms)[0]
        robot.leader_arms[leader].write("Torque_Enable", TorqueMode.DISABLED.value)
        print(f"MS3: Disabled {leader} Torque")
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
