from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass, field
from functools import cached_property
from typing import Mapping, cast

import numpy as np
import sapien
import sapien.physx as physx
import torch
from sapien.render import RenderCameraComponent

import mani_skill.render.utils as render_utils
from mani_skill.envs.utils.system.backend import parse_backend_device_id

# try and determine which render system is used by the installed sapien package
from mani_skill.render import SAPIEN_RENDER_SYSTEM
from mani_skill.sim.base_sim import BaseSim, BaseSimConfig
from mani_skill.sim.sapien.sensors.camera import SapienCamera
from mani_skill.sim.sapien.structs.actor import SapienActor
from mani_skill.sim.sapien.structs.articulation import SapienArticulation
from mani_skill.sim.sapien.structs.link import SapienLink
from mani_skill.sim.sapien.structs.render_camera import RenderCamera
from mani_skill.sim.sensors.base_sensor import BaseSensor
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.logging_utils import logger
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array

if SAPIEN_RENDER_SYSTEM == "3.1":
    from sapien.wrapper.scene import (
        get_camera_shader_pack,  # pyright: ignore[reportAttributeAccessIssue]
    )

    GlobalShaderPack = None
    sapien.render.RenderCameraGroup = "oldtype"


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
    actors: dict[str, SapienActor]
    articulations: dict[str, SapienArticulation]

    _sim_device: sapien.Device | None = None
    """The sapien device that the physics engine runs on."""
    _render_device: sapien.Device | None = None
    """The sapien device that the renderer runs on."""

    sub_scenes: list[sapien.Scene]
    """The list of SAPIEN sub-scenes."""
    px: physx.PhysxSystem
    """The physics system of the sub-scenes."""

    camera_groups: dict[str, sapien.render.RenderCameraGroup] = dict()  # type: ignore
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
            package_name, sim_backend, sim_device_id = parse_backend_device_id(
                sim_backend, sim_backend=True
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
                    parse_backend_device_id(render_backend, sim_backend=False)
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
        self._sensors_initialized = False
        self._human_render_cameras_initialized = False
        self._needs_fetch = False
        """
        Used internally to raise some errors ahead of time of when there may be
        undefined behaviors
        """
        super().__init__(
            num_envs,
            cfg,
            sim_device_torch=sim_device_torch,
            render_device_torch=render_device_torch,
        )
        if self.sim_device_torch.type == "cuda":
            if not physx.is_gpu_enabled():
                physx.enable_gpu()

        gpu_mem_config = self.cfg.gpu_memory_config.dict()

        # NOTE (stao): there isn't a easy way to check of collision_stack_size is supported for
        # the installed sapien3 version to get around that we just try and except. To be removed
        # once mac/windows platforms can upgrade to latest sapien versions
        try:
            physx.set_gpu_memory_config(**gpu_mem_config)
        except TypeError:
            gpu_mem_config.pop("collision_stack_size")
            physx.set_gpu_memory_config(**gpu_mem_config)

        sapien.render.set_log_level(os.getenv("MS_RENDERER_LOG_LEVEL", "warn"))
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

        builder = SapienActorBuilder()
        builder.sim = self
        return builder

    def create_articulation_builder(self):
        from mani_skill.sim.sapien.builders.articulation_builder import (
            SapienArticulationBuilder,
        )

        builder = SapienArticulationBuilder()
        builder.sim = self
        return builder

    def create_urdf_loader(self):
        from mani_skill.sim.sapien.loaders.urdf_loader import SapienURDFLoader

        builder = SapienURDFLoader()
        builder.sim = self
        return builder

    def remove_actor(self, actor: SapienActor):
        if self.gpu_sim_enabled:
            raise NotImplementedError(
                "Cannot remove actors after creating them in GPU sim at the moment"
            )
        else:
            self.sub_scenes[0].remove_entity(actor._objs[0])
            self.actors.pop(actor.name)

    def remove_articulation(self, articulation: SapienArticulation):
        if self.gpu_sim_enabled:
            raise NotImplementedError(
                "Cannot remove articulations after creating them in GPU sim at the moment"
            )
        else:
            entities = [link.entity for link in articulation._objs[0].links]
            for e in entities:
                self.sub_scenes[0].remove_entity(e)
            self.articulations.pop(articulation.name)

    @property
    def ambient_light(self):
        return self.sub_scenes[0].ambient_light

    @ambient_light.setter
    def ambient_light(self, color):
        for scene in self.sub_scenes:
            scene.render_system.ambient_light = color

    def set_ambient_light(self, color):
        self.ambient_light = color

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
        if position is None:
            position = [0, 0, 0]
        if scene_idxs is None:
            scene_idxs = list(range(len(self.sub_scenes)))
        for scene_idx in scene_idxs:
            if self.scene.parallel_in_single_scene:
                scene = self.sub_scenes[0]
            else:
                scene = self.sub_scenes[scene_idx]
            entity = sapien.Entity()
            entity.name = "directional_light"
            light = sapien.render.RenderDirectionalLightComponent()
            entity.add_component(light)
            light.color = color
            light.shadow = shadow
            light.shadow_near = shadow_near
            light.shadow_far = shadow_far
            light.shadow_half_size = shadow_scale
            light.shadow_map_size = shadow_map_size
            if self.scene.parallel_in_single_scene:
                light_position = position + self.scene_offsets_np[scene_idx]
            else:
                light_position = position
            light.pose = sapien.Pose(
                light_position,  # type: ignore
                sapien.math.shortest_rotation(
                    [1, 0, 0],  # type: ignore
                    direction,
                ),
            )
            scene.add_entity(entity)
            if self.scene.parallel_in_single_scene:
                break
        return

    def add_point_light(
        self,
        position,
        color,
        shadow=False,
        shadow_near=0.1,
        shadow_far=10.0,
        shadow_map_size=2048,
        scene_idxs: list[int] | None = None,
    ):
        if scene_idxs is None:
            scene_idxs = list(range(len(self.sub_scenes)))
        for scene_idx in scene_idxs:
            if self.scene.parallel_in_single_scene:
                scene = self.sub_scenes[0]
            else:
                scene = self.sub_scenes[scene_idx]
            entity = sapien.Entity()
            entity.name = "point_light"
            light = sapien.render.RenderPointLightComponent()
            entity.add_component(light)
            light.color = color
            light.shadow = shadow
            light.shadow_near = shadow_near
            light.shadow_far = shadow_far
            light.shadow_map_size = shadow_map_size
            if self.scene.parallel_in_single_scene:
                light.pose = sapien.Pose(position + self.scene_offsets_np[scene_idx])
            else:
                light.pose = sapien.Pose(position)

            scene.add_entity(entity)
        return light

    def add_area_light_for_ray_tracing(
        self,
        pose: sapien.Pose,
        color,
        half_width: float,
        half_height: float,
        scene_idxs=None,
    ):
        lighting_scenes = (
            self.sub_scenes
            if scene_idxs is None
            else [self.sub_scenes[i] for i in scene_idxs]
        )
        for scene in lighting_scenes:
            entity = sapien.Entity()
            light = sapien.render.RenderParallelogramLightComponent()
            entity.add_component(light)
            light.set_shape(half_width, half_height)
            light.color = color
            light.pose = pose
            scene.add_entity(entity)
        return

    def compile_render_scene(self):
        pass

    def add_camera(
        self,
        name: str,
        pose: Pose,
        width: int,
        height: int,
        near: float,
        far: float,
        fovy: float | list[float] | None = None,
        intrinsic: Array | None = None,
        mount: SapienActor | SapienLink | None = None,
    ) -> RenderCamera:
        if SAPIEN_RENDER_SYSTEM == "3.1":
            return self._sapien_31_add_camera(
                name, pose, width, height, near, far, fovy, intrinsic, mount
            )
        else:
            return self._sapien_add_camera(
                name, pose, width, height, near, far, fovy, intrinsic, mount
            )

    def _sapien_add_camera(
        self,
        name: str,
        pose: Pose,
        width: int,
        height: int,
        near: float,
        far: float,
        fovy: float | list[float] | None = None,
        intrinsic: Array | None = None,
        mount: SapienActor | SapienLink | None = None,
    ) -> RenderCamera:
        """internal helper function to add (mounted) cameras"""
        cameras = []
        pose = Pose.create(pose)
        # TODO (stao): support scene idxs property for cameras in the future
        # move intrinsic to np and batch intrinsic if it is not batched
        if intrinsic is not None:
            intrinsic = common.to_numpy(intrinsic)
            if len(intrinsic.shape) == 2:
                intrinsic = intrinsic[None, :]
                if len(self.sub_scenes) > 1:
                    # repeat the intrinsic along batch dim
                    intrinsic = intrinsic.repeat(len(self.sub_scenes), 0)
            assert len(intrinsic) == len(self.sub_scenes), (
                "intrinsic matrix batch dim not equal to the number of sub-scenes"
            )
        for i, scene in enumerate(self.sub_scenes):
            # Create camera component
            camera = RenderCameraComponent(width, height)
            if fovy is not None:
                if isinstance(fovy, float) or isinstance(fovy, int):
                    camera.set_fovy(fovy, compute_x=True)
                else:
                    camera.set_fovy(fovy[i], compute_x=True)
            elif intrinsic is not None:
                camera.set_focal_lengths(intrinsic[i, 0, 0], intrinsic[i, 1, 1])
                camera.set_principal_point(intrinsic[i, 0, 2], intrinsic[i, 1, 2])
            if isinstance(near, float) or isinstance(near, int):
                camera.near = near
            else:
                camera.near = near[i]
            if isinstance(far, float) or isinstance(far, int):
                camera.far = far
            else:
                camera.far = far[i]

            # mount camera to actor/link
            if mount is not None:
                if self.gpu_sim_enabled:
                    if isinstance(mount, SapienActor):
                        camera.set_gpu_pose_batch_index(
                            cast(
                                physx.PhysxRigidDynamicComponent,
                                mount._objs[i].find_component_by_type(
                                    physx.PhysxRigidDynamicComponent
                                ),
                            ).gpu_pose_index
                        )
                    elif isinstance(mount, SapienLink):
                        camera.set_gpu_pose_batch_index(mount._objs[i].gpu_pose_index)
                    else:
                        raise ValueError(
                            f"Tried to mount camera on object of type {mount.__class__}"
                        )
                if isinstance(mount, SapienLink):
                    mount._objs[i].entity.add_component(camera)
                else:
                    mount._objs[i].add_component(camera)
            else:
                camera_mount = sapien.Entity()
                camera_mount.add_component(camera)
                scene.add_entity(camera_mount)
                camera_mount.name = f"scene-{i}_{name}"
            if len(pose) == 1:
                camera.local_pose = pose.sp
            else:
                camera.local_pose = pose[i].sp
            camera.name = f"scene-{i}_{name}"
            cameras.append(camera)
        return RenderCamera.create(cameras, self, mount=mount)

    def _sapien_31_add_camera(
        self,
        name: str,
        pose: Pose,
        width: int,
        height: int,
        near: float,
        far: float,
        fovy: float | list[float] | None = None,
        intrinsic: Array | None = None,
        mount: SapienActor | SapienLink | None = None,
    ) -> RenderCamera:
        """internal helper function to add (mounted) cameras"""
        cameras = []
        pose = Pose.create(pose)
        # TODO (stao): support scene idxs property for cameras in the future
        # move intrinsic to np and batch intrinsic if it is not batched
        if intrinsic is not None:
            intrinsic = common.to_numpy(intrinsic)
            if len(intrinsic.shape) == 2:
                intrinsic = intrinsic[None, :]
                if len(self.sub_scenes) > 1:
                    # repeat the intrinsic along batch dim
                    intrinsic = intrinsic.repeat(len(self.sub_scenes), 0)
            assert len(intrinsic) == len(self.sub_scenes), (
                "intrinsic matrix batch dim not equal to the number of sub-scenes"
            )

        for i, scene in enumerate(self.sub_scenes):
            # Create camera component
            camera = RenderCameraComponent(
                width, height, GlobalShaderPack or get_camera_shader_pack()
            )
            if fovy is not None:
                if isinstance(fovy, (float, int)):
                    camera.set_fovy(fovy, compute_x=True)
                else:
                    camera.set_fovy(fovy[i], compute_x=True)
            elif intrinsic is not None:
                camera.set_focal_lengths(intrinsic[i, 0, 0], intrinsic[i, 1, 1])
                camera.set_principal_point(intrinsic[i, 0, 2], intrinsic[i, 1, 2])
            if isinstance(near, (float, int)):
                camera.near = near
            else:
                camera.near = near[i]
            if isinstance(far, (float, int)):
                camera.far = far
            else:
                camera.far = far[i]

            # mount camera to actor/link
            if mount is not None:
                if isinstance(mount, SapienLink):
                    mount._objs[i].entity.add_component(camera)
                else:
                    mount._objs[i].add_component(camera)
            else:
                camera_mount = sapien.Entity()
                camera_mount.set_pose(sapien.Pose([0, 0, 0]))
                camera_mount.add_component(camera)
                camera_mount.name = f"scene-{i}_{name}"
                scene.add_entity(camera_mount)
            if len(pose) == 1:
                camera.local_pose = pose.sp
            else:
                camera.local_pose = pose[i].sp
            camera.name = f"scene-{i}_{name}"
            cameras.append(camera)
            scene.update_render()
        return RenderCamera.create(cameras, self, mount=mount)

    def can_render(self):
        return True

    def update_render(
        self, update_sensors: bool = True, update_human_render_cameras: bool = True
    ):
        if SAPIEN_RENDER_SYSTEM == "3.1":
            self._sapien_31_update_render(
                update_sensors=update_sensors,
                update_human_render_cameras=update_human_render_cameras,
            )
        else:
            self._sapien_update_render(
                update_sensors=update_sensors,
                update_human_render_cameras=update_human_render_cameras,
            )

    def _sapien_update_render(
        self, update_sensors: bool = True, update_human_render_cameras: bool = True
    ):
        # note that this design is such that no GPU memory is allocated for memory unless
        # requested for, which can occur after the e.g. physx GPU simulation is initialized.
        if self.gpu_sim_enabled:
            if not self.scene.parallel_in_single_scene:
                if self.render_system_group is None:
                    self._setup_gpu_rendering()
                if not self._sensors_initialized and update_sensors:
                    self._gpu_setup_sensors(self.scene.sensors)
                    self._sensors_initialized = True
                if (
                    not self._human_render_cameras_initialized
                    and update_human_render_cameras
                ):
                    self._gpu_setup_sensors(self.scene.human_render_cameras)
                    self._human_render_cameras_initialized = True
                self.render_system_group.update_render()
            else:
                assert isinstance(self.px, physx.PhysxGpuSystem)
                self.px.sync_poses_gpu_to_cpu()
                self.sub_scenes[0].update_render()
        else:
            self.sub_scenes[0].update_render()

    def _sapien_31_update_render(
        self, update_sensors: bool = True, update_human_render_cameras: bool = True
    ):
        if self.gpu_sim_enabled:
            if self.render_system_group is None:
                for scene in self.sub_scenes:
                    scene.update_render()
                self._setup_gpu_rendering()
            if not self._sensors_initialized and update_sensors:
                self._gpu_setup_sensors(self.sensors)
                self._sensors_initialized = True
            if (
                not self._human_render_cameras_initialized
                and update_human_render_cameras
            ):
                self._gpu_setup_sensors(self.human_render_cameras)
                self._human_render_cameras_initialized = True

            manager: sapien.render.GpuSyncManager = (  # pyright: ignore[reportAttributeAccessIssue]
                self.render_system_group
            )
            manager.sync()
        else:
            self.sub_scenes[0].update_render()

    def compile_physical_scene(self):
        enable_gpu = self.gpu_sim_enabled
        if enable_gpu:
            assert isinstance(self.px, physx.PhysxGpuSystem)
            if SAPIEN_RENDER_SYSTEM == "3.1":
                for scene in self.sub_scenes:
                    scene.update_render()
            self.px.gpu_init()
        self.non_static_actors: list[SapienActor] = []
        # find non static actors, and set data indices that are now available after
        # gpu_init was called
        for actor in self.actors.values():
            if actor.body_type == "static":
                continue
            self.non_static_actors.append(actor)
            if enable_gpu:
                actor._body_data_index  # noqa only need to access this attribute to populate it

        for articulation in self.articulations.values():
            articulation._data_index  # noqa
            for link in articulation.links:
                link._body_data_index  # noqa

        for actor in self.non_static_actors:
            actor.set_pose(actor.initial_pose)
        for articulation in self.articulations.values():
            articulation.set_pose(articulation.initial_pose)

        if enable_gpu:
            assert isinstance(self.px, physx.PhysxGpuSystem)
            self.px.cuda_rigid_body_data.torch()[:, 7:] = torch.zeros_like(
                self.px.cuda_rigid_body_data.torch()[:, 7:]
            )  # zero out all velocities
            self.px.cuda_articulation_qvel.torch()[:, :] = torch.zeros_like(
                self.px.cuda_articulation_qvel.torch()
            )  # zero out all q velocities
            self.px.cuda_articulation_qf.torch()[:, :] = torch.zeros_like(
                self.px.cuda_articulation_qf.torch()
            )  # zero out all qf

            self.px.gpu_apply_rigid_dynamic_data()
            self.px.gpu_apply_articulation_root_pose()
            self.px.gpu_apply_articulation_root_velocity()
            self.px.gpu_apply_articulation_qvel()
            self.px.gpu_apply_articulation_qf()

            self._gpu_sim_initialized = True
            self.px.gpu_update_articulation_kinematics()
            self._gpu_fetch_all()

    def _gpu_apply_all(self):
        """
        Calls gpu_apply to update all body data, qpos, qvel, qf, and root poses
        """
        assert not self._needs_fetch, (
            "Once _gpu_apply_all is called, you must call _gpu_fetch_all before calling "
            "_gpu_apply_all again as otherwise there is undefined behavior that is likely "
            "impossible to debug"
        )
        assert isinstance(self.px, physx.PhysxGpuSystem)
        self.px.gpu_apply_rigid_dynamic_data()
        self.px.gpu_apply_articulation_qpos()
        self.px.gpu_apply_articulation_qvel()
        self.px.gpu_apply_articulation_qf()
        self.px.gpu_apply_articulation_root_pose()
        self.px.gpu_apply_articulation_root_velocity()
        self.px.gpu_apply_articulation_target_position()
        self.px.gpu_apply_articulation_target_velocity()
        self._needs_fetch = True

    def _gpu_fetch_all(self):
        """
        Queries simulation for all relevant GPU data. Note that this has some overhead.
        Should only be called at most once per simulation step as this automatically queries
        all data for all objects built in the scene.
        """
        assert isinstance(self.px, physx.PhysxGpuSystem)
        if len(self.non_static_actors) > 0:
            self.px.gpu_fetch_rigid_dynamic_data()

        if len(self.articulations) > 0:
            self.px.gpu_fetch_articulation_link_pose()
            self.px.gpu_fetch_articulation_link_velocity()
            self.px.gpu_fetch_articulation_qpos()
            self.px.gpu_fetch_articulation_qvel()
            self.px.gpu_fetch_articulation_qacc()
            self.px.gpu_fetch_articulation_target_qpos()
            self.px.gpu_fetch_articulation_target_qvel()

        self._needs_fetch = False

    def _gpu_update_articulation_kinematics(self):
        self.px.gpu_update_articulation_kinematics()  # type: ignore

    def _gpu_apply_articulation_target_position(self):
        self.px.gpu_apply_articulation_target_position()  # type: ignore

    def _gpu_apply_articulation_target_velocity(self):
        self.px.gpu_apply_articulation_target_velocity()  # type: ignore

    def physics_step(self):
        self.px.step()

    def can_physics(self):
        return True

    def get_pairwise_contact_impulses(
        self, obj1: SapienActor | SapienLink, obj2: SapienActor | SapienLink
    ):
        if self.gpu_sim_enabled:
            assert isinstance(self.px, physx.PhysxGpuSystem)
            query_hash = hash((obj1, obj2))
            query_key = obj1.name + obj2.name

            # we rebuild the potentially expensive contact query if it has not existed previously
            # or if it has, the managed objects are a different set
            rebuild_query = (query_key not in self._pairwise_contact_queries) or (
                query_key in self._pairwise_contact_query_unique_hashes
                and self._pairwise_contact_query_unique_hashes[query_key] != query_hash
            )
            if rebuild_query:
                body_pairs = cast(
                    list[
                        tuple[
                            physx.PhysxRigidBaseComponent, physx.PhysxRigidBaseComponent
                        ]
                    ],
                    list(zip(obj1._bodies, obj2._bodies)),
                )
                self._pairwise_contact_queries[query_key] = (
                    self.px.gpu_create_contact_pair_impulse_query(body_pairs)
                )
                self._pairwise_contact_query_unique_hashes[query_key] = query_hash

            query = self._pairwise_contact_queries[query_key]
            self.px.gpu_query_contact_pair_impulses(query)
            # query.cuda_impulses is shape (num_unique_pairs * num_envs, 3)
            pairwise_contact_impulses = query.cuda_impulses.torch().clone()
            return pairwise_contact_impulses
        else:
            assert isinstance(self.px, physx.PhysxCpuSystem)
            contacts = cast(physx.PhysxCpuSystem, self.px).get_contacts()
            pairwise_contact_impulses = sapien_utils.get_pairwise_contact_impulse(
                contacts, obj1._bodies[0].entity, obj2._bodies[0].entity
            )
            return common.to_tensor(pairwise_contact_impulses)[None, :]

    def get_contacts(self):
        if self.gpu_sim_enabled:
            raise NotImplementedError(
                "get_contacts is not available for GPU simulation"
            )
        else:
            assert isinstance(self.px, physx.PhysxCpuSystem)
            return self.px.get_contacts()

    ### GPU Simulation Management ###

    @cached_property
    def scene_offsets(self):
        """torch tensor of shape (num_envs, 3) representing the offset of each scene
        in the world frame"""

        if self.gpu_sim_enabled:
            assert isinstance(self.px, physx.PhysxGpuSystem)
            return torch.tensor(
                np.array(
                    [
                        self.px.get_scene_offset(sub_scene)
                        for sub_scene in self.sub_scenes
                    ]
                ),
                device=self.sim_device_torch,
            )
        else:
            raise NotImplementedError(
                "scene_offsets is not available for CPU simulation"
            )

    @cached_property
    def scene_offsets_np(self):
        """numpy array of shape (num_envs, 3) representing the offset of each scene in the
        world frame"""

        if self.gpu_sim_enabled:
            assert isinstance(self.px, physx.PhysxGpuSystem)
            return np.array(
                [self.px.get_scene_offset(sub_scene) for sub_scene in self.sub_scenes]
            )
        else:
            raise NotImplementedError(
                "scene_offsets_np is not available for CPU simulation"
            )

    ### CPU/GPU SAPIEN Rendering Code ###
    def _get_all_render_bodies(
        self,
    ) -> list[tuple[sapien.render.RenderBodyComponent, int]]:
        all_render_bodies = []
        for actor in self.actors.values():
            if actor.body_type == "static":
                continue
            all_render_bodies += [
                (
                    entity.find_component_by_type(sapien.render.RenderBodyComponent),
                    cast(
                        physx.PhysxRigidDynamicComponent,
                        entity.find_component_by_type(physx.PhysxRigidDynamicComponent),
                    ).gpu_pose_index,
                )
                for entity in actor._objs
            ]
        for articulation in self.articulations.values():
            all_render_bodies += [
                (
                    px_link.entity.find_component_by_type(
                        sapien.render.RenderBodyComponent
                    ),
                    px_link.gpu_pose_index,
                )
                for link in articulation.links
                for px_link in link._objs
            ]
        return all_render_bodies

    def _setup_gpu_rendering(self):
        if SAPIEN_RENDER_SYSTEM == "3.1":
            self._sapien_31_setup_gpu_rendering()
        else:
            self._sapien_setup_gpu_rendering()

    def _sapien_setup_gpu_rendering(self):
        """
        Prepares the scene for GPU parallelized rendering to enable taking e.g. RGB images
        """
        assert isinstance(self.px, physx.PhysxGpuSystem)
        for rb, gpu_pose_index in self._get_all_render_bodies():
            if rb is not None:
                for s in rb.render_shapes:
                    s.set_gpu_pose_batch_index(gpu_pose_index)
        self.render_system_group = sapien.render.RenderSystemGroup(
            [s.render_system for s in self.sub_scenes]
        )
        self.render_system_group.set_cuda_poses(self.px.cuda_rigid_body_data)

    def _sapien_31_setup_gpu_rendering(self):
        """
        Prepares the scene for GPU parallelized rendering to enable taking e.g. RGB images
        """
        assert isinstance(self.px, physx.PhysxGpuSystem)
        px = self.px

        shape_pose_indices = []
        shapes = []
        scene_id = 0
        for scene in self.sub_scenes:
            scene_id += 1
            for body in scene.render_system.render_bodies:
                b = body.entity.find_component_by_type(
                    sapien.physx.PhysxRigidBodyComponent
                )
                if b is None:
                    continue
                for s in body.render_shapes:
                    shape_pose_indices.append(
                        b.gpu_pose_index  # pyright: ignore[reportAttributeAccessIssue]
                    )
                    shapes.append(s)

        cam_pose_indices = []
        cams = []
        for cameras in self.scene.sensors.values():
            assert isinstance(cameras, SapienCamera), (
                f"Expected SapienCamera, got {cameras}"
            )
            for c in cameras.camera._render_cameras:
                b = c.entity.find_component_by_type(
                    sapien.physx.PhysxRigidBodyComponent
                )
                if b is None:
                    continue
                cam_pose_indices.append(b.gpu_pose_index)
                cams.append(c)

        sync_manager = (
            sapien.render.GpuSyncManager()  # pyright: ignore[reportAttributeAccessIssue]
        )
        sync_manager.set_cuda_poses(px.cuda_rigid_body_data)
        sync_manager.set_render_shapes(shape_pose_indices, shapes)
        sync_manager.set_cameras(cam_pose_indices, cams)

        self.render_system_group = sync_manager

    def _gpu_setup_sensors(self, sensors: Mapping[str, BaseSensor]):
        if SAPIEN_RENDER_SYSTEM == "3.1":
            self._sapien_31_gpu_setup_sensors(sensors)
        else:
            self._sapien_gpu_setup_sensors(sensors)

    def _sapien_gpu_setup_sensors(self, sensors: Mapping[str, BaseSensor]):
        for name, sensor in sensors.items():
            if isinstance(sensor, SapienCamera):
                try:
                    assert self.render_system_group is not None
                    camera_group = self.render_system_group.create_camera_group(
                        sensor.camera._render_cameras,
                        list(sensor.config.shader_config.texture_names.keys()),
                    )
                except RuntimeError as e:
                    raise RuntimeError(
                        "Unable to create GPU parallelized camera group. "
                        "If the error is about being unable to create a buffer, you are "
                        "likely using too many Cameras. Either use less cameras (via less "
                        "parallel envs) and/or reduce the size of the cameras. Another common "
                        "cause is using a memory intensive shader. You can try using the "
                        "'minimal' shader which optimizes for GPU memory but disables some "
                        "advanced functionalities. Another option is to avoid rendering with the "
                        "rgb_array mode or using the human render cameras, as they can be more "
                        "memory intensive (they typically have higher resolutions for the purposes "
                        "of visualization)."
                    ) from e

                sensor.camera.camera_group = camera_group
                self.camera_groups[name] = camera_group
            else:
                raise NotImplementedError(
                    f"This sensor {sensor} of type {sensor.__class__} has not been "
                    "implemented yet on the GPU"
                )

    def _sapien_31_gpu_setup_sensors(self, sensors: Mapping[str, BaseSensor]):
        for name, sensor in sensors.items():
            if isinstance(sensor, SapienCamera):
                batch_renderer = sapien.render.RenderManager(  # pyright: ignore[reportAttributeAccessIssue]
                    sapien.render.get_shader_pack(  # pyright: ignore[reportAttributeAccessIssue]
                        sensor.config.shader_config.shader_pack
                    )
                )
                batch_renderer.set_size(sensor.config.width, sensor.config.height)
                batch_renderer.set_cameras(sensor.camera._render_cameras)
                sensor.camera.camera_group = self.camera_groups[name] = batch_renderer
            else:
                raise NotImplementedError(
                    f"This sensor {sensor} of type {sensor.__class__} has not been "
                    "implemented yet on the GPU"
                )
