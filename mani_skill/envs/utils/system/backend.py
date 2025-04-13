"""
Utilities for determining the simulation backend and devices
"""
import platform
from dataclasses import dataclass

import sapien
import torch

from mani_skill.utils.logging_utils import logger


@dataclass
class BackendInfo:
    device: torch.device
    """the device in which to return all simulation data on"""
    sim_device: sapien.Device
    """the device on which the physics simulation is running"""
    sim_backend: str
    """the backend name of the physics simulation"""
    render_device: sapien.Device
    """the device on which the renderer is running"""
    render_backend: str
    """the backend name of the renderer"""


CPU_SIM_BACKENDS = set(["cpu", "physx_cpu"])

sim_backend_name_mapping = {
    "cpu": "physx_cpu",
    "cuda": "physx_cuda",
    "gpu": "physx_cuda",
    "physx_cpu": "physx_cpu",
    "physx_cuda": "physx_cuda",
}
render_backend_name_mapping = {
    "cpu": "sapien_cpu",
    "cuda": "sapien_cuda",
    "gpu": "sapien_cuda",
    "sapien_cpu": "sapien_cpu",
    "sapien_cuda": "sapien_cuda",
}


def parse_sim_and_render_backend(sim_backend: str, render_backend: str) -> BackendInfo:
    sim_backend = sim_backend_name_mapping[sim_backend]
    render_backend = render_backend_name_mapping[render_backend]
    if sim_backend == "physx_cpu":
        device = torch.device("cpu")
        sim_device = sapien.Device("cpu")
    elif sim_backend == "physx_cuda":
        device = torch.device("cuda")
        sim_device = sapien.Device("cuda")
    elif sim_backend[:4] == "cuda":
        device = torch.device(sim_backend)
        sim_device = sapien.Device(sim_backend)
    else:
        raise ValueError(f"Invalid simulation backend: {sim_backend}")

    if platform.system() == "Darwin":
        render_device = sapien.Device("cpu")
        render_backend = "sapien_cpu"
        logger.warning(
            "Detected MacOS system, forcing render backend to be sapien_cpu and render device to be MacOS compatible."
        )
    elif render_backend == "sapien_cuda":
        render_device = sapien.Device("cuda")
    elif render_backend == "sapien_cpu":
        render_device = sapien.Device("cpu")
    elif render_backend[:4] == "cuda":
        render_device = sapien.Device(render_backend)
    else:
        # handle special cases such as for AMD gpus, render_backend must be defined as pci:... instead as cuda is not available.
        render_device = sapien.Device(render_backend)
    return BackendInfo(
        device=device,
        sim_device=sim_device,
        sim_backend=sim_backend_name_mapping[sim_backend],
        render_device=render_device,
        render_backend=render_backend,
    )
