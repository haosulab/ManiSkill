"""
Various sampling functions/classes for fast, vectorized sampling of e.g. object poses
"""

from typing import Tuple

import torch

from mani_skill.utils import common
from mani_skill.utils.geometry.rotation_conversions import Device


class UniformPlacementSampler:
    """Uniform placement sampler that lets you sequentially sample data such that the data is within given bounds and
    not too close to previously sampled data. This sampler is also batched so you can use this easily for GPU simulated tasks

    Args:
        bounds: ((low1, low2, ...), (high1, high2, ...))
        batch_size (int): The number of points to sample with each call to sample(...)
    """

    def __init__(
        self,
        bounds: Tuple[list[float], list[float]],
        batch_size: int,
        device: Device = None,
    ) -> None:
        assert len(bounds) == 2 and len(bounds[0]) == len(bounds[1])
        self._bounds = common.to_tensor(bounds, device=device)
        self._ranges = self._bounds[1] - self._bounds[0]
        self.fixtures_radii = None
        self.fixture_positions = None
        self.batch_size = batch_size

    def sample(self, radius, max_trials, append=True, verbose=False):
        """Sample a position.

        Args:
            radius (float): collision radius.
            max_trials (int): maximal trials to sample.
            append (bool, optional): whether to append the new sample to fixtures. Defaults to True.
            verbose (bool, optional): whether to print verbosely. Defaults to False.

        Returns:
            torch.Tensor: a sampled position.
        """
        if self.fixture_positions is None:
            sampled_pos = (
                torch.rand((self.batch_size, self._bounds.shape[1])) * self._ranges
                + self._bounds[0]
            )
        else:
            pass_mask = torch.zeros((self.batch_size), dtype=bool)
            sampled_pos = torch.zeros((self.batch_size, self._bounds.shape[1]))
            for i in range(max_trials):
                pos = (
                    torch.rand((self.batch_size, self._bounds.shape[1])) * self._ranges
                    + self._bounds[0]
                )  # (B, d)
                dist = torch.linalg.norm(
                    pos - self.fixture_positions, axis=-1
                )  # (n, B)
                radii = self.fixtures_radii + radius  # (n, )
                mask = torch.all(dist > radii[:, None], axis=0)  # (B, )
                sampled_pos[mask] = pos[mask]
                pass_mask[mask] = True
                if torch.all(pass_mask):
                    if verbose:
                        print(
                            f"Found valid set of {self.batch_size=} samples at {i}-th trial"
                        )
                    break
            else:
                if verbose:
                    print("Fail to find a valid sample!")
        if append:
            if self.fixture_positions is None:
                self.fixture_positions = sampled_pos[None, ...]
            else:
                self.fixture_positions = torch.concat(
                    [self.fixture_positions, sampled_pos[None, ...]]
                )
            if self.fixtures_radii is None:
                self.fixtures_radii = common.to_tensor(radius).reshape(
                    1,
                )
            else:
                self.fixtures_radii = torch.concat(
                    [
                        self.fixtures_radii,
                        common.to_tensor(radius).reshape(
                            1,
                        ),
                    ]
                )
        return sampled_pos
