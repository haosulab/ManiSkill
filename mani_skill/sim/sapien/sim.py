import platform
from dataclasses import asdict, dataclass, field

import numpy as np
import sapien
import sapien.physx as physx
import torch

import mani_skill.render.utils as render_utils
from mani_skill.sim.base_sim import BaseSim, BaseSimConfig
from mani_skill.utils.logging_utils import logger
from mani_skill.utils.structs.pose import Pose


@dataclass
class GPUMemoryConfig:
    """A gpu memory configuration dataclass that neatly holds all parameters that configure physx
    GPU memory for simulation"""

    temp_buffer_capacity: int = 2**24
    """Increase this if you get 'PxgPinnedHostLinearMemoryAllocator: overflowing initial allocation
    size, increase capacity to at least %.' """
    max_rigid_contact_count: int = 2**19
    """Increase this if you get 'Contact buffer overflow detected'"""
    max_rigid_patch_count: int = (
        2**18
    )  # 81920 is SAPIEN default but most tasks work with 2**18
    """Increase this if you get 'Patch buffer overflow detected'"""
    heap_capacity: int = 2**26
    found_lost_pairs_capacity: int = (
        2**25
    )  # 262144 is SAPIEN default but most tasks use 2**25
    found_lost_aggregate_pairs_capacity: int = 2**10
    total_aggregate_pairs_capacity: int = 2**10
    collision_stack_size: int = 64 * 64 * 1024  # default as SAPIEN
    """Increase this if you get 'Collision stack overflow detected'"""

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


@dataclass
class SceneConfig:
    gravity: np.ndarray | list[float] = field(
        default_factory=lambda: np.array([0, 0, -9.81])
    )
    bounce_threshold: float = 2.0
    sleep_threshold: float = 0.005
    contact_offset: float = 0.02
    rest_offset: float = 0
    solver_position_iterations: int = 15
    solver_velocity_iterations: int = 1
    enable_pcm: bool = True
    enable_tgs: bool = True
    enable_ccd: bool = False
    enable_enhanced_determinism: bool = False
    enable_friction_every_iteration: bool = True
    cpu_workers: int = 0

    def dict(self):
        return {k: v for k, v in asdict(self).items()}

    # cpu_workers=min(os.cpu_count(), 4)
    # NOTE (fxiang): PCM is enabled for GPU sim regardless.
    # NOTE (fxiang): smaller contact_offset is faster as less contacts are considered, but some
    # contacts may be missed if distance changes too fast
    # NOTE (fxiang): solver iterations 15 is recommended to balance speed and accuracy. If stable
    # grasps are necessary >= 20 is preferred.
    # NOTE (fxiang): can try using more cpu_workers as it may also make it faster if there are a lot
    # of collisions, collision filtering is on CPU
    # NOTE (fxiang): enable_enhanced_determinism is for CPU probably. If there are 10 far apart sub
    # scenes, this being True makes it so they do not impact each other at all


@dataclass(frozen=True)
class SapienSimConfig(BaseSimConfig):
    gpu_memory_config: GPUMemoryConfig = field(default_factory=GPUMemoryConfig)
    scene_config: SceneConfig = field(default_factory=SceneConfig)


class SapienSim(BaseSim):
    # TODO (stao): sapien sim uses sim_backend and render_backend strings, but BaseSim(ABC) uses 
    # devices. Consolidate the two?
    """
    Simulation backend for SAPIEN.

    Args:
        num_envs: The number of environments to simulate.
        cfg: The configuration for the simulation backend.
        sim_backend: The backend to use for the simulation. If none,
            this sim object is not performing any physics simulation.
        render_backend: The backend to use for the rendering. If none,
            this sim object is not performing any rendering.
    """
    id: str = "sapien"
    cfg: SapienSimConfig

    _sim_device: sapien.Device | None = None
    """The sapien device that the physics engine runs on."""
    _render_device: sapien.Device | None = None
    """The sapien device that the renderer runs on."""

    sub_scenes: list[sapien.Scene]
    """The list of SAPIEN sub-scenes."""
    px: physx.PhysxSystem
    """The physics system of the sub-scenes."""

    camera_groups: dict[str, sapien.render.RenderCameraGroup] = dict()
    """The Sapien camera groups of the sub-scenes for tiled rendering"""
    render_system_group: sapien.render.RenderSystemGroup | None = None
    """The Sapien render system group of the sub-scenes for tiled rendering"""

    def __init__(
        self,
        num_envs: int = 1,
        cfg: SapienSimConfig | None = None,
        sim_backend: str | None = "sapien.physx_cpu",
        render_backend: str | None = "sapien.cuda",
    ):
        if cfg is None:
            cfg = SapienSimConfig()

        # Determine devices simulation and/or rendering are running on
        sim_device_torch = torch.device("cpu")
        render_device_torch = torch.device("cpu")
        if sim_backend is not None:
            package_name, sim_backend, sim_device_id = self._parse_backend_device_id(
                sim_backend
            )
            assert package_name == "sapien"
            if sim_backend == "physx_cpu":
                sim_device_torch = torch.device("cpu")
                self._sim_device = sapien.Device("cpu")
            elif sim_backend == "physx_cuda":
                device_str = (
                    f"cuda:{sim_device_id}" if sim_device_id is not None else "cuda"
                )
                sim_device_torch = torch.device(device_str)
                self._sim_device = sapien.Device(device_str)
            elif sim_backend[:4] == "cuda":
                device_str = (
                    f"cuda:{sim_device_id}" if sim_device_id is not None else "cuda"
                )
                sim_device_torch = torch.device(device_str)
                self._sim_device = sapien.Device(device_str)
            else:
                raise ValueError(f"Invalid simulation backend: {sim_backend}")

        try:
            if render_backend is not None:
                package_name, render_backend, render_device_id = (
                    self._parse_backend_device_id(render_backend)
                )
                assert package_name == "sapien"
                if platform.system() == "Darwin":
                    self._render_device = sapien.Device("cpu")
                    render_device_torch = torch.device("cpu")
                    render_backend = "sapien_cpu"
                    logger.warning(
                        "Detected MacOS system, forcing render backend to be sapien:cpu in order "
                        "to be MacOS compatible."
                    )
                elif render_backend == "sapien_cuda":
                    device_str = (
                        f"cuda:{render_device_id}"
                        if render_device_id is not None
                        else "cuda"
                    )
                    self._render_device = sapien.Device(device_str)
                    render_device_torch = torch.device(device_str)
                elif render_backend == "sapien_cpu":
                    self._render_device = sapien.Device("cpu")
                    render_device_torch = torch.device("cpu")
                elif render_backend[:4] == "cuda":
                    device_str = (
                        f"cuda:{render_device_id}"
                        if render_device_id is not None
                        else "cuda"
                    )
                    self._render_device = sapien.Device(device_str)
                    render_device_torch = torch.device(device_str)
                elif render_backend == "none" or render_backend is None:
                    self._render_device = None
                    render_device_torch = torch.device("cpu")
                else:
                    # handle special cases such as for AMD gpus, render_backend must be defined as
                    # pci:... instead as cuda is not available.
                    self._render_device = sapien.Device(render_backend)
                    render_device_torch = torch.device(render_backend)
        except RuntimeError as e:
            if str(e) == 'failed to find device "cuda"':
                logger.warning(
                    f'Requested to use render device "{render_backend}", but CUDA device was not '
                    'found. Falling back to "cpu" device. Rendering might be disabled.'
                )
                self._render_device = sapien.Device("cpu")
                render_device_torch = torch.device("cpu")
                render_backend = "sapien_cpu"
            else:
                raise e

        super().__init__(
            num_envs,
            cfg,
            physics_device_torch=sim_device_torch,
            render_device_torch=render_device_torch,
        )

        self._set_scene_config()
        self._build_sub_scenes()

    def _set_scene_config(self):
        """
        Set Sapien scene configuration.
        """
        physx.set_shape_config(
            contact_offset=self.cfg.scene_config.contact_offset,
            rest_offset=self.cfg.scene_config.rest_offset,
        )
        physx.set_body_config(
            solver_position_iterations=self.cfg.scene_config.solver_position_iterations,
            solver_velocity_iterations=self.cfg.scene_config.solver_velocity_iterations,
            sleep_threshold=self.cfg.scene_config.sleep_threshold,
        )
        gravity = self.cfg.scene_config.gravity
        if not isinstance(gravity, np.ndarray):
            gravity = np.array(gravity)
        physx.set_scene_config(
            gravity=gravity,  # pyright: ignore[reportArgumentType]
            bounce_threshold=self.cfg.scene_config.bounce_threshold,
            enable_pcm=self.cfg.scene_config.enable_pcm,
            enable_tgs=self.cfg.scene_config.enable_tgs,
            enable_ccd=self.cfg.scene_config.enable_ccd,
            enable_enhanced_determinism=self.cfg.scene_config.enable_enhanced_determinism,
            enable_friction_every_iteration=self.cfg.scene_config.enable_friction_every_iteration,  # noqa: E501
            cpu_workers=self.cfg.scene_config.cpu_workers,
        )
        physx.set_default_material(**self.cfg.default_materials_config.dict())

    def _build_sub_scenes(self):
        if self._sim_device is not None and self._sim_device.is_cuda():
            physx_system = physx.PhysxGpuSystem(device=self._sim_device)
            # Create the scenes in a square grid
            sub_scenes = []
            scene_grid_length = int(np.ceil(np.sqrt(self.num_envs)))
            for scene_idx in range(self.num_envs):
                scene_x, scene_y = (
                    scene_idx % scene_grid_length - scene_grid_length // 2,
                    scene_idx // scene_grid_length - scene_grid_length // 2,
                )
                systems: list[sapien.System] = [physx_system]
                if render_utils.can_render(self._render_device):
                    systems.append(sapien.render.RenderSystem(self._render_device))
                scene = sapien.Scene(systems=systems)
                physx_system.set_scene_offset(
                    scene,
                    [
                        scene_x * self.cfg.spacing,
                        scene_y * self.cfg.spacing,
                        0,
                    ],
                )
                sub_scenes.append(scene)
        else:
            physx_system = physx.PhysxCpuSystem()
            systems = [physx_system]
            if render_utils.can_render(self._render_device):
                systems.append(sapien.render.RenderSystem(self._render_device))
            sub_scenes = [sapien.Scene(systems)]

        self.sub_scenes = sub_scenes
        self.px = self.sub_scenes[0].physx_system
        assert all(
            isinstance(s.physx_system, type(self.px)) for s in self.sub_scenes
        ), "all sub-scenes must use the same simulation backend"
        self.px.timestep = 1.0 / self.cfg.sim_freq

        # TODO (stao): do all sims need this property?
        self.gpu_sim_enabled = (
            True if isinstance(self.px, physx.PhysxGpuSystem) else False
        )
        """whether the sub scenes are using the GPU or CPU backend"""
        self._gpu_sim_initialized = False
        """whether the GPU simulation has been initialized"""
        self._pairwise_contact_queries: dict[
            str, physx.PhysxGpuContactPairImpulseQuery
        ] = dict()
        """dictionary mapping pairwise contact query keys to GPU contact queries. Used in GPU
        simulation only to cache queries as query creation will pause any GPU sim computation"""
        self._pairwise_contact_query_unique_hashes: dict[str, int] = dict()
        """maps keys in self.pairwise_contact_queries to unique hashes dependent on the actual
        objects involved in the query. This is used to determine automatically when to rebuild
        contact queries as keys for self.pairwise_contact_queries are kept non-unique between
        episode resets in order to be easily rebuilt and deallocate old queries. This essentially
        acts as a way to invalidate the cached queries."""

    def create_actor_builder(self):
        from mani_skill.sim.sapien.builders.actor_builder import SapienActorBuilder

        return SapienActorBuilder().set_scene(self)

    def create_articulation_builder(self):
        from mani_skill.sim.sapien.builders.articulation_builder import (
            SapienArticulationBuilder,
        )

        return SapienArticulationBuilder().set_scene(self)

    ### Code for compiling simulator scene for rendering ###
    def compile_render_scene(self):
        pass

    def add_camera(self, pose: Pose):
        pass

    def can_render(self):
        return True

    ### Code for compiling simulator scene for physical simulation ###
    def compile_physical_scene(self):
        pass

    def physics_step(self):
        pass

    def can_physics(self):
        return True
