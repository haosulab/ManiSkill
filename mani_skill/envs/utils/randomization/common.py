from typing import Sequence, Union

import torch

from mani_skill.utils.structs.types import Device


def uniform(
    low: Union[float, torch.Tensor],
    high: Union[float, torch.Tensor],
    size: Sequence,
    device: Device = None,
):
    dist = high - low
    return torch.rand(size=size, device=device) * dist + low
