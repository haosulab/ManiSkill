from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.base_sensor import BaseSensorConfig
from mani_skill.utils.structs.types import Array


class BaseRealAgent:
    """
    Base agent class for representing real robots, sensors, and controlling them in a real environment. This generally should be used with the :py:class:`mani_skill.envs.sim2real_env.Sim2RealEnv` class for deploying policies learned in simulation
    to the real world.

    Args:
        sensor_configs (Dict[str, BaseSensorConfig]): the sensor configs to create the agent with.
    """

    def __init__(self, sensor_configs: Dict[str, BaseSensorConfig] = dict()):
        self.sensor_configs = sensor_configs

        self._sim_agent: BaseAgent = None

        @dataclass
        class RealRobot:
            agent: BaseRealAgent

            @property
            def qpos(self):
                return self.get_qpos()

            def get_qpos(self):
                return self.agent.get_qpos()

            @property
            def qvel(self):
                return self.get_qvel()

            def get_qvel(self):
                return self.agent.get_qvel()

        self.robot = RealRobot(self)
        """
        a reference to a fake robot/articulation used for accessing qpos/qvel values.
        """

    @property
    def controller(self):
        return self._sim_agent.controller

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
    # functions for controlling the agent
    # ---------------------------------------------------------------------------- #
    def set_target_qpos(self, qpos: Array):
        """
        Set the target joint positions of the agent.
        Args:
            qpos: the joint positions in radians to set the agent to.
        """
        # equivalent to set_drive_targets in simulation
        raise NotImplementedError

    def set_target_qvel(self, qvel: Array):
        """
        Set the target joint velocities of the agent.
        Args:
            qvel: the joint velocities in radians/s to set the agent to.
        """
        # equivalent to set_drive_velocity_targets in simulation
        raise NotImplementedError

    def reset(self, qpos: Array):
        """
        Reset the agent to a given qpos. For real robots this function should move the robot at a safe and controlled speed to the given qpos and aim to reach it accurately.
        Args:
            qpos: the qpos in radians to reset the agent to.
        """
        raise NotImplementedError

    # ---------------------------------------------------------------------------- #
    # data access for e.g. joint position values, sensor observations etc.
    # All of the def get_x() functions should return numpy arrays and be implemented
    # ---------------------------------------------------------------------------- #
    def capture_sensor_data(self, sensor_names: Optional[List[str]] = None):
        """
        Capture the sensor data asynchronously from the agent based on the given sensor names. If sensor_names is None then all sensor data should be captured. This should not return anything and should be async if possible.
        """
        raise NotImplementedError

    def get_sensor_data(self, sensor_names: Optional[List[str]] = None):
        """
        Get the desired sensor observations from the agent based on the given sensor names. If sensor_names is None then all sensor data should be returned. The expected format for cameras is in line with the simulation's
        format for cameras.

        .. code-block:: python

            {
                "sensor_name": {
                    "rgb": torch.uint8 (1, H, W, 3), # red green blue image colors
                    "depth": torch.int16 (1, H, W, 1), # depth in millimeters
                }
            }

        whether rgb or depth is included depends on the real camera and can be omitted if not supported or not used. Note that a batch dimension is expected in the data.

        For more details see https://maniskill.readthedocs.io/en/latest/user_guide/concepts/sensors.html in order to ensure
        the real data aligns with simulation formats.
        """
        raise NotImplementedError

    def get_sensor_params(self, sensor_names: List[str] = None):
        """
        Get the parameters of the desired sensors based on the given sensor names. If sensor_names is None then all sensor parameters should be returned. The expected format for cameras is in line with the simulation's
        format is:

        .. code-block:: python

            {
                "sensor_name": {
                    "cam2world_gl": [4, 4], # transformation from the camera frame to the world frame (OpenGL/Blender convention)
                    "extrinsic_cv": [4, 4], # camera extrinsic (OpenCV convention)
                        "intrinsic_cv": [3, 3], # camera intrinsic (OpenCV convention)
                }
            }


        If these numbers are not needed/unavailable it is okay to leave the fields blank. Some observation processing modes may need these fields however such as point clouds in the world frame.
        """
        return dict()

    def get_qpos(self):
        """
        Get the current joint positions in radians of the agent as a torch tensor. Data should have a batch dimension, the shape should be (1, N) for N joint positions.
        """
        raise NotImplementedError(
            "This real agent does not an implementation for getting joint positions. If you do not need joint positions and are using the Sim2Real env setup you can override the simulation _get_obs_agent function to remove the qpos information."
        )

    def get_qvel(self):
        """
        Get the current joint velocities in radians/s of the agent as a torch tensor. Data should have a batch dimension, the shape should be (1, N) for N joint velocities.
        """
        raise NotImplementedError(
            "This real agent does not an implementation for getting joint velocities. If you do not need joint velocities and are using the Sim2Real env setup you can override the simulation _get_obs_agent function to remove the qvel information."
        )

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

    def get_proprioception(self):
        """
        Get the proprioceptive state of the agent, default is the qpos and qvel of the robot and any controller state.

        Note that if qpos or qvel functions are not implemented they will return None.
        """
        obs = dict(qpos=self.get_qpos(), qvel=self.get_qvel())
        controller_state = self._sim_agent.controller.get_state()
        if len(controller_state) > 0:
            obs.update(controller=controller_state)
        return obs

    def __getattr__(self, name):
        """
        Delegate attribute access to self._sim_agent if the attribute doesn't exist in self.
        This allows accessing sim_agent properties and methods directly from the real agent. Some simulation agent include convenience functions to access e.g. end-effector poses
        or various properties of the robot.
        """
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            if hasattr(self, "_sim_agent") and hasattr(self._sim_agent, name):
                return getattr(self._sim_agent, name)
            raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")
