from dataclasses import dataclass, field

import numpy as np


@dataclass
class MPMSystemConfig:
    """
    Configuration for the MPM system
    """

    gravity: np.ndarray = field(default_factory=lambda: np.array([0, 0, -9.81]))
    """gravity acceleration in m/s^2, defaults to -9.81"""
    sim_freq: int = 100
    """simulation frequency (Hz), 1 / sim_freq is the time step of each simulation step"""
