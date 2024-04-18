from typing import Sequence

import torch

from mani_skill.utils.structs.types import Device


def uniform(low: float, high: float, size: Sequence, device: Device = None):
    dist = high - low
    return torch.rand(size=size, device=device) * dist + low
