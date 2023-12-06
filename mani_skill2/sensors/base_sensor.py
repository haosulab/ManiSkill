from gymnasium import spaces


class BaseSensor:
    """
    Base class for all sensors
    """

    def __init__(self, sensor_type: str) -> None:
        self.sensor_type = sensor_type

    def setup(self):
        """
        Setup this sensor given the current scene. This is called during environment/scene reconfiguration.
        """
        raise NotImplementedError()

    def pre_get_obs(self):
        """
        Override this function for any code to call after a simulation step.

        Some sensors like rgbd cameras need to take a picture just once after each call to scene.update_render
        """
        pass

    def get_obs(self):
        raise NotImplementedError()

    def get_params(self):
        """
        Get parameters for this sensor
        """
        raise NotImplementedError()

    @property
    def observation_space(self) -> spaces.Space:
        raise NotImplementedError()


class BaseSensorConfig:
    # TODO (stao):
    pass
