"""
Utilities for generating randomized values like poses as well as useful samplers to sample e.g. object positions without collision 
"""

import numpy as np
import transforms3d


def random_quaternion(rng: np.random.RandomState):
    # Uniform sample a quaternion
    q = transforms3d.quaternions.axangle2quat(rng.rand(3), 2 * np.pi * rng.rand())
    return q


def random_position(rng: np.random.RandomState):
    pass


def generate_random_reachable_position(rng: np.random.RandomState, source, max_dist):
    pass
