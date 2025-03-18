import torch

from mani_skill.utils.structs.types import Array


class Builder:
    def add_particles(self, particles: Array):
        """
        Directly add particles to this entity

        Args:
            particles: (N, 3) array of particles determining the position of each particle in this entity
        """
