"""
Code implementation for a batched random number generator. The goal is to enable seeding a batched random number generator with a batch of seeds to ensure randomization
in CPU simulators and GPU simulators are the same
"""

from typing import List, Union

import numpy as np


class BatchedRNG(np.random.RandomState):
    def __init__(self, rngs: List):
        self.rngs = rngs
        self.batch_size = len(rngs)

    @classmethod
    def from_seeds(cls, seeds: List[int], backend: str = "numpy:random_state"):
        batch_size = len(seeds)
        if backend == "numpy:random_state":
            return cls(rngs=[np.random.RandomState(seed) for seed in seeds])
        raise ValueError(f"Unknown batched RNG backend: {backend}")

    @classmethod
    def from_rngs(cls, rngs: List):
        return cls(rngs=rngs)

    def __getitem__(self, idx: Union[int, List[int], np.ndarray]):
        if np.iterable(idx):
            return BatchedRNG.from_rngs([self.rngs[i] for i in idx])
        return self.rngs[idx]

    def __getattribute__(self, item):
        if item in [
            "rngs",
            "__class__",
            "__dict__",
            "__getattribute__",
            "__str__",
            "__repr__",
            "__getitem__",
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
