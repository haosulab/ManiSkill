"""
Utilities for determining the simulation backend and devices
"""

from dataclasses import dataclass

from mani_skill.utils.logging_utils import logger


@dataclass
class BackendInfo:
    sim_backend_package: str
    """the package name of the physics simulation backend"""
    sim_backend: str
    """the full backend name of the physics simulation"""
    render_backend_package: str
    """the package name of the renderer"""
    render_backend: str
    """the full backend name of the renderer"""


CPU_SIM_BACKENDS = set(["cpu", "physx_cpu", "sapien:physx_cpu"])

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


def parse_backend_device_id(
    backend: str, sim_backend: bool = True
) -> tuple[str, str, str | None]:
    if "." in backend:
        package_name, backend_name = backend.split(".")
        parts = backend_name.split(":")
        if len(parts) == 2:
            return package_name, parts[0], parts[1]
        return package_name, backend_name, None
    else:
        # Backward compatability for old backend format
        logger.warning(
            f"Using legacy backend naming: {backend}. Please use the new format "
            "<package_name.backend_name> instead."
        )

        if sim_backend:
            if backend == "physx_cpu":
                return "sapien", "physx_cpu", None
            elif backend == "physx_cuda":
                return "sapien", "physx_cuda", None
            elif backend == "gpu":
                return "sapien", "physx_cuda", None
            elif backend == "cuda":
                return "sapien", "physx_cuda", None
            elif backend == "cpu":
                return "sapien", "physx_cpu", None
        else:
            if backend == "sapien_cpu":
                return "sapien", "sapien_cpu", None
            elif backend == "sapien_cuda":
                return "sapien", "sapien_cuda", None
            elif backend == "cuda":
                return "sapien", "sapien_cuda", None
            elif backend == "gpu":
                return "sapien", "sapien_cuda", None
            elif backend == "cpu":
                return "sapien", "sapien_cpu", None
    raise ValueError(
        f"Invalid backend: {backend}. Should be in the format "
        "<package_name.backend_name> or <package_name.backend_name:device_id>."
    )


def parse_sim_and_render_backend(sim_backend: str, render_backend: str) -> BackendInfo:
    # Parse sim_backend string to check for CUDA device specification
    package_name, sim_backend, sim_device_id = parse_backend_device_id(
        sim_backend, sim_backend=True
    )
    package_name, render_backend, render_device_id = parse_backend_device_id(
        render_backend, sim_backend=False
    )
    return BackendInfo(
        # device=device,
        # sim_device=sim_device,
        sim_backend_package=package_name,
        sim_backend=f"{package_name}.{sim_backend}"
        + (f":{sim_device_id}" if sim_device_id is not None else ""),
        render_backend_package=package_name,
        render_backend=f"{package_name}.{render_backend}"
        + (f":{render_device_id}" if render_device_id is not None else ""),
    )
