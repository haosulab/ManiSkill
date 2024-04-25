from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
from gymnasium import spaces

from .base_controller import BaseController, ControllerConfig


class PassiveController(BaseController):
    """
    Passive controller that does not do anything
    """

    config: "PassiveControllerConfig"

    def set_drive_property(self):
        n = len(self.joints)
        damping = np.broadcast_to(self.config.damping, n)
        force_limit = np.broadcast_to(self.config.force_limit, n)
        friction = np.broadcast_to(self.config.friction, n)

        for i, joint in enumerate(self.joints):
            joint.set_drive_properties(0, damping[i], force_limit=force_limit[i])
            joint.set_friction(friction[i])

    def _initialize_action_space(self):
        self.single_action_space = spaces.Box(
            np.empty(0), np.empty(0), dtype=np.float32
        )

    def set_action(self, action: np.ndarray):
        pass

    def before_simulation_step(self):
        pass


@dataclass
class PassiveControllerConfig(ControllerConfig):
    damping: Union[float, Sequence[float]]
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0
    controller_cls = PassiveController
