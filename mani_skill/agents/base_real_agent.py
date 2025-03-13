from typing import Dict

import numpy as np

from mani_skill.sensors.base_sensor import BaseSensorConfig


class BaseRealAgent:
    """
    Base agent class for representing real robots, sensors, and controlling them in a real environment.

    Args:
        sensor_configs (Dict[str, BaseSensorConfig]): the sensor configs to create the agent with.
    """

    def __init__(self, sensor_configs: Dict[str, BaseSensorConfig]):
        self.sensor_configs = sensor_configs

    # ---------------------------------------------------------------------------- #
    # functions for controlling the agent
    # ---------------------------------------------------------------------------- #
    def set_target_qpos(self, qpos: np.ndarray):
        """
        Set the target joint positions of the agent.
        Args:
            qpos (np.ndarray): the joint positions to set the agent to.
        """
        # equivalent to set_drive_targets in simulation
        raise NotImplementedError

    def set_target_qvel(self, qvel: np.ndarray):
        """
        Set the target joint velocities of the agent.
        Args:
            qvel (np.ndarray): the joint velocities to set the agent to.
        """
        # equivalent to set_drive_velocity_targets in simulation
        raise NotImplementedError

    def reset(self, qpos: np.ndarray):
        """
        Reset the agent to a given qpos. For real robots this function should move the robot at a safe and controlled speed to the given qpos and aim to reach it accurately.
        Args:
            qpos (np.ndarray): the qpos to reset the agent to.
        """
        raise NotImplementedError

    def start(self):
        """
        Start the agent, which include turning on the motors/robot, setting up cameras/sensors etc.

        For sensors you have access to self.sensor_configs which is the requested sensor setup. For e.g. cameras these sensor configs will define the camera resolution.

        For sim2real/real2sim alignment when defining real environment interfaces we instantiate the real agent with the simulation environment's sensor configs.
        """
        raise NotImplementedError

    def stop(self):
        """
        Stop the agent, which include turning off the motors/robot, stopping cameras/sensors etc.
        """
        raise NotImplementedError

    # ---------------------------------------------------------------------------- #
    # data access for e.g. joint position values, sensor observations etc.
    # All of the def get_x() functions should return numpy arrays and be implemented
    # ---------------------------------------------------------------------------- #
    def capture_sensor_data(self):
        """
        Capture the sensor data asynchronously from the agent. This should not return anything.
        """
        raise NotImplementedError

    def get_sensor_obs(self):
        """
        Get the sensor observations from the agent. The expected format for most cameras is

        ```
        {
            "sensor_name": {
                "rgb": np.uint8 (H, W, 3), # red green blue image colors
                "depth": np.int16 (H, W, 1), # depth in millimeters
            }
        }
        ```

        whether rgb or depth is included depends on the real camera and can be omitted if not supported.

        For more details see https://maniskill.readthedocs.io/en/latest/user_guide/concepts/sensors.html in order to ensure
        the real data aligns with simulation formats.
        """
        raise NotImplementedError

    def get_qpos(self):
        """
        Get the current joint positions of the agent.
        """
        raise NotImplementedError

    def get_qvel(self):
        """
        Get the current joint velocities of the agent.
        """
        raise NotImplementedError

    @property
    def qpos(self):
        """
        Get the current joint positions of the agent.
        """
        return self.get_qpos()

    @property
    def qvel(self):
        """
        Get the current joint velocities of the agent.
        """
        return self.get_qvel()
