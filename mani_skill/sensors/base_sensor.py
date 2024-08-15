from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class BaseSensorConfig:
    uid: str


class BaseSensor:
    """
    Base class for all sensors
    """

    def __init__(self, config: BaseSensorConfig) -> None:
        self.config = config

    def setup(self) -> None:
        """
        Setup this sensor given the current scene. This is called during environment/scene reconfiguration.
        """

    def capture(self) -> None:
        """
        Captures sensor data and prepares it for it to be then retrieved via get_obs for observations and get_image for a visualizable image.

        Some sensors like rgbd cameras need to take a picture just once after each call to scene.update_render. Generally this should also be a
        non-blocking function if possible.
        """

    def get_obs(self, **kwargs):
        """
        Retrieves captured sensor data as an observation for use by an agent.
        """
        raise NotImplementedError()

    def get_params(self) -> Dict:
        """
        Get parameters for this sensor. Should return a dictionary with keys mapping to torch.Tensor values
        """
        raise NotImplementedError()

    def get_images(self) -> torch.Tensor:
        """
        This returns the data of the sensor visualized as an image (rgb array of shape (B, H, W, 3)). This should not be used for generating agent observations. For example lidar data can be visualized
        as an image but should not be in a image format (H, W, 3) when being used by an agent.
        """
        raise NotImplementedError()

    @property
    def uid(self):
        return self.config.uid
