"""
Code implementation for a batched random number generator. The goal is to enable seeding a batched random number generator with a batch of seeds to ensure randomization
in CPU simulators and GPU simulators are the same
"""
import numpy as np


class BatchedRNG(np.random.RandomState):
    def __init__(self, seeds: list[int]):
        self.rngs = [np.random.RandomState(seed) for seed in seeds]
        self.batch_size = len(seeds)

    def __getattribute__(self, item):
        if item in [
            "rngs",
            "__class__",
            "__dict__",
            "__getattribute__",
            "__str__",
            "__repr__",
            "batch_size",
        ]:
            return object.__getattribute__(self, item)
        if callable(getattr(self.rngs[0], item)):

            def method(*args, **kwargs):
                return np.array(
                    [
                        object.__getattribute__(rng, item)(*args, **kwargs)
                        for rng in self.rngs
                    ]
                )

            return method
        else:
            return np.array([getattr(rng, item) for rng in self.rngs])
