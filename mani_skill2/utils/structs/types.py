from typing import Sequence, Union

import numpy as np
import sapien.physx as physx
import torch


def get_backend_name():
    if physx.is_gpu_enabled():
        return "torch"
    else:
        return "numpy"


Array = Union[torch.Tensor, np.array, Sequence]
Device = Union[str, torch.device]
