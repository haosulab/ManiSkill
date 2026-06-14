from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

import torch

from mani_skill.sim.builders.actor import BaseActorBuilder
from mani_skill.sim.builders.articulation import BaseArticulationBuilder
from mani_skill.utils.structs.pose import Pose


@dataclass(frozen=True)
class DefaultMaterialsConfig:
    # note these frictions are same as unity
    static_friction: float = 0.3
    dynamic_friction: float = 0.3
    restitution: float = 0

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


@dataclass(frozen=True)
class BaseSimConfig:
    """
    Base configuration dataclass for the simulation backends.
    """

    spacing: float = 5.0
    """Controls the spacing between parallel environments when simulating on GPU in meters.
    Increase this value if you expect objects in one parallel environment to impact objects
    within this spacing distance."""
    sim_freq: int = 120
    """simulation frequency (Hz)."""
    control_freq: int = 60
    """control frequency (Hz). Every control step (e.g. env.step) contains
    (sim_freq / control_freq) physics steps."""

    default_materials_config: DefaultMaterialsConfig = field(
        default_factory=DefaultMaterialsConfig
    )


class BaseSim(ABC):
    """
    Base class for all simulation backends.

    A simulation backend consists of primarily a physics engine and a renderer. It is possible
    for a simulation backend to only have one or the other as well.
    """

    id: str
    """The id of the simulation backend."""
    physics_device_torch: torch.device
    """The torch device that physics engine returns data on."""
    render_device_torch: torch.device
    """The torch device that the renderer returns data on."""
    cfg: BaseSimConfig
    """The configuration for the simulation backend."""
    num_envs: int
    """The number of environments to simulate."""
    _reset_mask: torch.Tensor
    """A mask for controlling which sub-scenes permit modifications to object data"""

    def __init__(
        self,
        num_envs: int = 1,
        cfg: BaseSimConfig | None = None,
        physics_device_torch: torch.device | None = None,
        render_device_torch: torch.device | None = None,
    ):
        if physics_device_torch is None:
            physics_device_torch = torch.device("cpu")
        if render_device_torch is None:
            render_device_torch = torch.device("cpu")
        self.num_envs = num_envs
        self.cfg = cfg or BaseSimConfig()
        self.physics_device_torch = physics_device_torch
        self.render_device_torch = render_device_torch

    def _parse_backend_device_id(self, backend: str) -> tuple[str, str, str | None]:
        if "." in backend:
            package_name, backend_name = backend.split(".")
            parts = backend_name.split(":")
            if len(parts) == 2:
                return package_name, parts[0], parts[1]
            return package_name, backend_name, None
        raise ValueError(
            f"Invalid backend: {backend}. Should be in the format "
            "<package_name.backend_name> or <package_name.backend_name:device_id>."
        )

    ### Shared derived properties ###
    @property
    def timestep(self) -> float:
        """The timestep of the simulation."""
        return 1.0 / self.cfg.sim_freq

    ### Code for adding builders to a scene for rendering/physics simulation ###
    @abstractmethod
    def create_actor_builder(self) -> BaseActorBuilder:
        """
        Creates an ActorBuilder object that can be used to build actors in this scene.
        """

    @abstractmethod
    def create_articulation_builder(self) -> BaseArticulationBuilder:
        """
        Creates an ArticulationBuilder object that can be used to build articulations in
        this scene.
        """

    ### Code for compiling simulator scene for rendering ###
    @abstractmethod
    def compile_render_scene(self):
        """
        Compiles the simulation scene for rendering.
        """

    ### Rendering code ###
    # TODO (stao): add_camera or call this add_sensor and eventually support other kinds of sensors?
    # feel like cameras need a lot of special treatment in general...
    # (e.g. mounting, batching+tiling, evals etc.)
    @abstractmethod
    def add_camera(self, pose: Pose):
        """
        Adds a camera to the simulation scene.
        """

    @abstractmethod
    def can_render(self):
        """
        Whether the simulation backend can render.
        """

    ### Code for compiling simulator scene for physical simulation ###
    @abstractmethod
    def compile_physical_scene(self):
        """
        Compiles the simulation scene for physical simulation. Usually necessary to have an
        explicit compilation stage for simulators with GPU parallelization, but some
        simulators permit larger changes to the physical scene at runtime.
        """

    ### Physical simulation code ###
    @abstractmethod
    def physics_step(self):
        """
        Runs a single physics step at `self.cfg.sim_freq` Hz.
        """

    @abstractmethod
    def can_physics(self):
        """
        Whether the simulation backend can run physical simulation.
        """
