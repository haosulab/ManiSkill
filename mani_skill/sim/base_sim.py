from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import torch

from mani_skill.sim.builders.actor import BaseActorBuilder
from mani_skill.sim.builders.articulation import BaseArticulationBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.render_camera import RenderCamera
from mani_skill.utils.structs.types import Array

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene


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

    Args:
        num_envs: The number of environments to simulate.
        cfg: The configuration for the simulation backend.
        physics_device_torch: The torch device that physics engine returns data on. If none,
            this sim object is not performing any physics simulation.
        render_device_torch: The torch device that the renderer returns data on. If none,
            this sim object is not performing any rendering.
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
    scene: ManiSkillScene
    """The ManiSkillScene that this simulation backend is associated with."""
    gpu_sim_enabled: bool
    """Whether the simulation backend is batched."""
    actors: dict[str, Actor]
    """The dictionary of actors in the simulation backend."""
    articulations: dict[str, Articulation]
    """The dictionary of articulations in the simulation backend."""

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
        if self.physics_device_torch.type == "cuda":
            self.gpu_sim_enabled = True
        else:
            self.gpu_sim_enabled = False
        self.actors = dict()
        self.articulations = dict()

    def _parse_backend_device_id(self, backend: str) -> tuple[str, str, str | None]:
        if "." in backend:
            package_name, backend_name = backend.split(".")
            parts = backend_name.split(":")
            if len(parts) == 2:
                return package_name, parts[0], parts[1]
            return package_name, backend_name, None
        else:
            # Backward compatability for old backend format
            if backend == "physx_cpu":
                return "sapien", "physx_cpu", None
            elif backend == "physx_cuda":
                return "sapien", "physx_cuda", None
            elif backend == "cuda":
                return "sapien", "cuda", None
            elif backend == "cpu":
                return "sapien", "cpu", None
            elif backend == "sapien_cuda":
                return "sapien", "sapien_cuda", None
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

    def remove_actor(self, actor: Actor):
        """
        Removes an actor from the simulation scene.
        """
        raise NotImplementedError()

    def remove_articulation(self, articulation: Articulation):
        """
        Removes an articulation from the simulation scene.
        """
        raise NotImplementedError()

    ### Code for working with cameras and sensors ###
    def add_camera(
        self,
        name: str,
        pose: Pose,
        width: int,
        height: int,
        near: float,
        far: float,
        fovy: float | list[float],
        intrinsic: Array | None = None,
        mount: Actor | Link | None = None,
    ) -> RenderCamera:
        """
        Adds a camera to the simulation scene.
        """
        raise NotImplementedError()

    ### Code for lighting ###
    def add_directional_light(
        self,
        direction,
        color,
        shadow=False,
        position=None,
        shadow_scale=10.0,
        shadow_near=-10.0,
        shadow_far=10.0,
        shadow_map_size=2048,
        scene_idxs: list[int] | None = None,
    ):
        raise NotImplementedError()

    ### Code for compiling simulator scene for rendering ###
    @abstractmethod
    def compile_render_scene(self):
        """
        Compiles the simulation scene for rendering.
        """

    ### Rendering code ###
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

    ### Accelerate data management code ###

    def _gpu_apply_all(self):
        """
        Calls gpu_apply to update all body data, qpos, qvel, qf, and root poses
        """
        raise NotImplementedError()

    def _gpu_fetch_all(self):
        """
        Queries simulation for all relevant GPU data. Note that this has some overhead.
        Should only be called at most once per simulation step as this automatically queries
        all data for all objects built in the scene.
        """
        raise NotImplementedError()

    def _gpu_update_articulation_kinematics(self):
        """
        Updates the articulation kinematics on the GPU.
        """
        raise NotImplementedError()
