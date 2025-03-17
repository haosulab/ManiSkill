import gymnasium as gym
import numpy as np
from lerobot.common.robot_devices.cameras.configs import (
    IntelRealSenseCameraConfig,
    OpenCVCameraConfig,
)
from lerobot.common.robot_devices.motors.configs import DynamixelMotorsBusConfig
from lerobot.common.robot_devices.robots.configs import KochRobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

import mani_skill.envs.tasks.digital_twins.koch_arm.pickcube
from mani_skill.agents.robots.lerobot.manipulator import LeRobotRealAgent
from mani_skill.envs.sim2real_env import Sim2RealEnv
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

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
        "base_camera": IntelRealSenseCameraConfig(
            serial_number=146322070293,
            fps=30,
            width=640,
            height=480,
        ),
    },
    calibration_dir="koch_calibration",
)
real_robot = ManipulatorRobot(robot_config)

# max control freq for lerobot really is just 60Hz
real_agent = LeRobotRealAgent(real_robot)


wrappers = [FlattenRGBDObservationWrapper]
sim_env = gym.make(
    "KochPickCubeEnv-v1",
    obs_mode="rgb",
    sim_config={"sim_freq": 120, "control_freq": 30},
)
for wrapper in wrappers:
    sim_env = wrapper(sim_env)
real_env = Sim2RealEnv(sim_env=sim_env, agent=real_agent, obs_mode="rgb")
sim_env.print_sim_details()
sim_obs, _ = sim_env.reset()
real_obs, _ = real_env.reset()

for k in sim_obs.keys():
    print(
        f"{k}: sim_obs shape: {sim_obs[k].shape}, real_obs shape: {real_obs[k].shape}"
    )

done = False
while not done:
    action = real_env.action_space.sample() * 0.7
    real_obs, _, terminated, truncated, info = real_env.step(action)
    done = terminated or truncated
sim_env.close()
real_agent.stop()
