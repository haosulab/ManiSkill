from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

import sapien
import sapien.render
import torch

from mani_skill.envs.utils.system.backend import BackendInfo
from mani_skill.sim.sensors.base_sensor import BaseSensor
from mani_skill.sim.sensors.camera import Camera
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.render_camera import RenderCamera
from mani_skill.utils.structs.types import Array, Device, SimConfig

if TYPE_CHECKING:
    from mani_skill.sim.base_sim import BaseSim


@dataclass
class StateDictRegistry:
    actors: dict[str, Actor]
    articulations: dict[str, Articulation]


class ManiSkillScene:
    """
    ManiSkillScene class manages the core simulation and all parallel sub-scenes without any
    simulator backend specific code.
    """

    def __init__(
        self,
        physics_sim: BaseSim,
        render_sim: BaseSim,
        sim_config: SimConfig | None = None,
        device: Optional[Device] = None,
        parallel_in_single_scene: bool = False,
        backend: Optional[BackendInfo] = None,
    ):
        assert device is not None, "device argument is required"
        assert backend is not None, "backend argument is required"
        if sim_config is None:
            sim_config = SimConfig()
        self.gpu_sim_enabled = physics_sim.gpu_sim_enabled
        self.physics_sim = physics_sim
        self.render_sim = render_sim
        self.physics_sim.scene = self
        self.render_sim.scene = self
        self._shared_sim_packages = self.physics_sim == self.render_sim
        # TODO (stao): optimizations if physics and render sims are the same object
        # e.g. both using sapien

        self.sim_config = sim_config

        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.backend = backend  # references the backend object stored in BaseEnv class

        self.actor_views: dict[str, Actor] = dict()
        """views of actors in any sub-scenes created by using Actor.merge and queryable as if it
        were a single Actor"""

        self.articulation_views: dict[str, Articulation] = dict()
        """views of articulations in any sub-scenes created by using Articulation.merge and
        queryable as if it were a single Articulation"""

        self.sensors: dict[str, BaseSensor] = dict()
        self.human_render_cameras: dict[str, Camera] = dict()
        self._sensors_initialized = False
        self._human_render_cameras_initialized = False

        self._reset_mask = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        """
        Used internally by various objects like Actor, Link, and Controllers to auto mask out
        sub-scenes so they do not get modified during partial env resets
        """

        self.parallel_in_single_scene: bool = parallel_in_single_scene
        """Whether rendering all parallel scenes in the viewer/gui is enabled"""

        self.state_dict_registry: StateDictRegistry = StateDictRegistry(
            actors=dict(), articulations=dict()
        )
        """state dict registry that map actor/articulation names to Actor/Articulation struct
        references. Only these structs are used for the environment state"""

    @property
    def actors(self):
        return self.physics_sim.actors

    @property
    def articulations(self):
        return self.physics_sim.articulations

    @property
    def _gpu_sim_initialized(self) -> bool:
        """whether the GPU simulation has been initialized"""
        # TODO (stao): some functions might only care if the render sim is initialized...?
        return self.physics_sim._gpu_sim_initialized

    def can_render(self):
        """
        Whether or not this Scene object permits rendering, depending on the rendering device
        selected
        """

        return self.render_sim.can_render()

    # -------------------------------------------------------------------------- #
    # Functions from sapien.Scene
    # -------------------------------------------------------------------------- #

    def create_actor_builder(self):
        """Creates an ActorBuilder object that can be used to build actors in this scene."""
        from mani_skill.sim.builders.actor import BaseActorBuilder

        builder = BaseActorBuilder()
        if self._shared_sim_packages:
            builder._add_sim(self.physics_sim)
        else:
            builder._add_sim(self.physics_sim)
            builder._add_sim(self.render_sim)
        return builder

    def create_articulation_builder(self):
        """Creates an ArticulationBuilder object that can be used to build articulations in this
        scene."""

        from mani_skill.sim.builders.articulation import BaseArticulationBuilder

        builder = BaseArticulationBuilder()
        if self._shared_sim_packages:
            builder._add_sim(self.physics_sim)
        else:
            builder._add_sim(self.physics_sim)
            builder._add_sim(self.render_sim)
        return builder

    def create_urdf_loader(self):
        """Creates a URDFLoader object that can be used to load URDF files into this scene"""
        from mani_skill.sim.loaders.urdf import BaseURDFLoader

        loader = BaseURDFLoader()
        if self._shared_sim_packages:
            loader._add_sim(self.physics_sim)
        else:
            loader._add_sim(self.physics_sim)
            loader._add_sim(self.render_sim)
        return loader

    def create_mjcf_loader(self):
        """Creates a MJCFLoader object that can be used to load MJCF files into this scene"""
        from ..utils.building.mjcf_loader import MJCFLoader

        loader = MJCFLoader()
        loader.set_scene(self)
        return loader

    # def create_physical_material(
    #     self, static_friction: float, dynamic_friction: float, restitution: float
    # ):
    #     return physx.PhysxMaterial(static_friction, dynamic_friction, restitution)

    def remove_actor(self, actor: Actor):
        """Removes an actor from the scene. Only works in CPU simulation."""
        if self._shared_sim_packages:
            self.physics_sim.remove_actor(actor)
        else:
            self.physics_sim.remove_actor(actor)
            self.render_sim.remove_actor(actor)

    def remove_articulation(self, articulation: Articulation):
        """Removes an articulation from the scene. Only works in CPU simulation."""
        if self._shared_sim_packages:
            self.physics_sim.remove_articulation(articulation)
        else:
            self.physics_sim.remove_articulation(articulation)
            self.render_sim.remove_articulation(articulation)

    def add_camera(
        self,
        name,
        pose,
        width,
        height,
        near,
        far,
        fovy: float | list[float] | None = None,
        intrinsic: Array | None = None,
        mount: Actor | Link | None = None,
    ) -> RenderCamera:
        """Adds a (mounted) camera to the scene"""
        return self.render_sim.add_camera(
            name, pose, width, height, near, far, fovy, intrinsic, mount
        )

    # def remove_camera(self, camera):
    #     self.remove_entity(camera.entity)

    # def get_cameras(self):
    #     return self.render_system.cameras

    # def get_mounted_cameras(self):
    #     return self.get_cameras()

    def step(self):
        self.physics_sim.physics_step()

    def get_contacts(self):
        self.physics_sim.get_contacts()

    def get_all_actors(self):
        """
        Returns a list of all sapien.Entity objects that have rigid dynamic and static components
        across all sub scenes.
        """

        return [
            c.entity
            for c in self.px.rigid_dynamic_components + self.px.rigid_static_components
        ]

    def get_all_articulations(self):
        """
        Returns list of all physx articulation objects across all sub scenes
        """
        return [
            c.articulation for c in self.px.articulation_link_components if c.is_root
        ]

    def create_drive(
        self,
        body0: Union[Actor, Link],
        pose0: Union[sapien.Pose, Pose],
        body1: Union[Actor, Link],
        pose1: Union[sapien.Pose, Pose],
    ):
        from mani_skill.sim.sapien.structs.drive import Drive

        # body0 and body1 should be in parallel.
        return Drive.create_from_actors_or_links(
            self.physics_sim, body0, pose0, body1, pose1, body0._scene_idxs
        )

    # def create_connection(
    #     self,
    #     body0: Optional[Union[sapien.Entity, physx.PhysxRigidBaseComponent]],
    #     pose0: sapien.Pose,
    #     body1: Union[sapien.Entity, physx.PhysxRigidBaseComponent],
    #     pose1: sapien.Pose,
    # ):
    #     if body0 is None:
    #         c0 = None
    #     elif isinstance(body0, sapien.Entity):
    #         c0 = next(
    #             c
    #             for c in body0.components
    #             if isinstance(c, physx.PhysxRigidBaseComponent)
    #         )
    #     else:
    #         c0 = body0

    #     assert body1 is not None
    #     if isinstance(body1, sapien.Entity):
    #         e1 = body1
    #         c1 = next(
    #             c
    #             for c in body1.components
    #             if isinstance(c, physx.PhysxRigidBaseComponent)
    #         )
    #     else:
    #         e1 = body1.entity
    #         c1 = body1

    #     connection = physx.PhysxDistanceJointComponent(c1)
    #     connection.parent = c0
    #     connection.pose_in_child = pose1
    #     connection.pose_in_parent = pose0
    #     e1.add_component(connection)
    #     connection.set_limit(0, 0)
    #     return connection

    # def create_gear(
    #     self,
    #     body0: Optional[Union[sapien.Entity, physx.PhysxRigidBaseComponent]],
    #     pose0: sapien.Pose,
    #     body1: Union[sapien.Entity, physx.PhysxRigidBaseComponent],
    #     pose1: sapien.Pose,
    # ):
    #     if body0 is None:
    #         c0 = None
    #     elif isinstance(body0, sapien.Entity):
    #         c0 = next(
    #             c
    #             for c in body0.components
    #             if isinstance(c, physx.PhysxRigidBaseComponent)
    #         )
    #     else:
    #         c0 = body0

    #     assert body1 is not None
    #     if isinstance(body1, sapien.Entity):
    #         e1 = body1
    #         c1 = next(
    #             c
    #             for c in body1.components
    #             if isinstance(c, physx.PhysxRigidBaseComponent)
    #         )
    #     else:
    #         e1 = body1.entity
    #         c1 = body1

    #     gear = physx.PhysxGearComponent(c1)
    #     gear.parent = c0
    #     gear.pose_in_child = pose1
    #     gear.pose_in_parent = pose0
    #     e1.add_component(gear)
    #     return gear

    # @property
    # def render_id_to_visual_name(self):
    #     # TODO
    #     return

    # def set_environment_map(self, cubemap: str):
    #     if isinstance(cubemap, str):
    #         self.render_system.cubemap = sapien.render.RenderCubemap(cubemap)
    #     else:
    #         self.render_system.cubemap = cubemap

    # def set_environment_map_from_files(
    #     self, px: str, nx: str, py: str, ny: str, pz: str, nz: str
    # ):
    #     self.render_system.cubemap = sapien.render.RenderCubemap(px, nx, py, ny, pz, nz)

    # ---------------------------------------------------------------------------- #
    # Additional useful properties / functions
    # ---------------------------------------------------------------------------- #
    @property
    def num_envs(self):
        return self.physics_sim.num_envs

    def get_pairwise_contact_impulses(self, obj1: Actor | Link, obj2: Actor | Link):
        """
        Get the impulse vectors between two actors/links. Returns impulse vector of shape
        (N, 3), where N is the number of environments and 3 is the dimension of the impulse
        vector itself, representing x, y, and z direction of impulse.

        Note that dividing the impulse value by self.px.timestep yields the pairwise contact
        force in Newtons. The equivalent API for that is self.get_pairwise_contact_force(obj1,
        obj2). It is generally recommended to use the force values since they are independent of
        the timestep (dt = 1 / sim_freq) of the simulation.

        Args:
            obj1: Actor | Link
            obj2: Actor | Link
        """
        return self.physics_sim.get_pairwise_contact_impulses(obj1, obj2)

    def get_pairwise_contact_forces(
        self, obj1: Union[Actor, Link], obj2: Union[Actor, Link]
    ):
        """
        Get the force vectors between two actors/links. Returns force vector of shape (N, 3)
        where N is the number of environments and 3 is the dimension of the force vector itself,
        representing x, y, and z direction of force.

        Args:
            obj1: Actor | Link
            obj2: Actor | Link
        """
        return (
            self.get_pairwise_contact_impulses(obj1, obj2) / self.physics_sim.timestep
        )

    # -------------------------------------------------------------------------- #
    # Simulation state (required for MPC)
    # -------------------------------------------------------------------------- #

    def add_to_state_dict_registry(self, object: Union[Actor, Articulation]):
        if isinstance(object, Actor):
            assert object.name not in self.state_dict_registry.actors, (
                f"Object {object.name} already in state dict registry"
            )
            self.state_dict_registry.actors[object.name] = object
        elif isinstance(object, Articulation):
            assert object.name not in self.state_dict_registry.articulations, (
                f"Object {object.name} already in state dict registry"
            )
            self.state_dict_registry.articulations[object.name] = object
        else:
            raise ValueError(f"Expected Actor or Articulation, got {object}")

    def remove_from_state_dict_registry(self, object: Union[Actor, Articulation]):
        if isinstance(object, Actor):
            assert object.name in self.state_dict_registry.actors, (
                f"Object {object.name} not in state dict registry"
            )
            del self.state_dict_registry.actors[object.name]
        elif isinstance(object, Articulation):
            assert object.name in self.state_dict_registry.articulations, (
                f"Object {object.name} not in state dict registry"
            )
            del self.state_dict_registry.articulations[object.name]
        else:
            raise ValueError(f"Expected Actor or Articulation, got {object}")

    def get_sim_state(self) -> dict[str, dict[str, torch.Tensor]]:
        """Get simulation state.

        Returns a dictionary with two nested dictionaries, "actors" and "articulations".
        In the nested dictionaries, each maps the actor/articulation name to a vector of shape
        (N, D) where N is the number of parallel environments and D is the dimension of the
        padded state per environment.

        Note that static actor data are not included. It is expected that an environment
        reconstructs itself deterministically such that the same static actors always have the
        same states.
        """

        state_dict = dict()
        state_dict["actors"] = dict()
        state_dict["articulations"] = dict()
        for actor in self.state_dict_registry.actors.values():
            if actor.body_type == "static":
                continue
            state_dict["actors"][actor.name] = actor.get_state().clone()
        for articulation in self.state_dict_registry.articulations.values():
            state_dict["articulations"][articulation.name] = (
                articulation.get_state().clone()
            )
        if len(state_dict["actors"]) == 0:
            del state_dict["actors"]
        if len(state_dict["articulations"]) == 0:
            del state_dict["articulations"]
        return state_dict

    def set_sim_state(self, state: dict, env_idx: Optional[torch.Tensor] = None):
        if env_idx is not None:
            prev_reset_mask = self._reset_mask.clone()
            # safe guard against setting the wrong states
            self._reset_mask[:] = False
            self._reset_mask[env_idx] = True

        if "actors" in state:
            for actor_id, actor_state in state["actors"].items():
                if len(actor_state.shape) == 1:
                    actor_state = actor_state[None, :]
                # do not pass in env_idx to avoid redundant reset mask changes
                self.state_dict_registry.actors[actor_id].set_state(actor_state, None)
        if "articulations" in state:
            for art_id, art_state in state["articulations"].items():
                if len(art_state.shape) == 1:
                    art_state = art_state[None, :]
                self.state_dict_registry.articulations[art_id].set_state(
                    art_state, None
                )
        if env_idx is not None:
            self._reset_mask = prev_reset_mask

    # ---------------------------------------------------------------------------- #
    # GPU Simulation Management
    # ---------------------------------------------------------------------------- #
    def _setup(self):
        """
        Start the CPU/GPU simulation and allocate all buffers and initialize objects
        """
        self.physics_sim.compile_physical_scene()
        self.render_sim.compile_render_scene()

    def _gpu_apply_all(self):
        """
        Calls gpu_apply to update all body data, qpos, qvel, qf, and root poses
        """
        if self._shared_sim_packages:
            self.physics_sim._gpu_apply_all()
        else:
            self.physics_sim._gpu_apply_all()
            self.render_sim._gpu_apply_all()

    def _gpu_fetch_all(self):
        """
        Queries simulation for all relevant GPU data. Note that this has some overhead.
        Should only be called at most once per simulation step as this automatically queries
        all data for all objects built in the scene.
        """
        if self.gpu_sim_enabled:
            if self._shared_sim_packages:
                self.physics_sim._gpu_fetch_all()
            else:
                self.physics_sim._gpu_fetch_all()
                self.render_sim._gpu_fetch_all()

    def _gpu_update_articulation_kinematics(self):
        # NOTE (stao): this is a bit specific to physx/sapien I think
        if self._shared_sim_packages:
            self.physics_sim._gpu_update_articulation_kinematics()
        else:
            self.physics_sim._gpu_update_articulation_kinematics()
            self.render_sim._gpu_update_articulation_kinematics()

    # ---------------------------------------------------------------------------- #
    # CPU/GPU sim Rendering Code
    # ---------------------------------------------------------------------------- #

    def get_sensor_images(
        self, obs: dict[str, Any]
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Get raw sensor data as images for visualization purposes."""
        sensor_data = dict()
        for name, sensor in self.sensors.items():
            sensor_data[name] = sensor.get_images(obs[name])
        return sensor_data

    def get_human_render_camera_images(
        self, camera_name: Optional[str] = None
    ) -> dict[str, torch.Tensor]:
        image_data = dict()
        if self.gpu_sim_enabled:
            if self.parallel_in_single_scene:
                for name, camera in self.human_render_cameras.items():
                    camera.camera._render_cameras[0].take_picture()
                    rgb = camera.get_obs(
                        rgb=True, depth=False, segmentation=False, position=False
                    )["rgb"]
                    image_data[name] = rgb
            else:
                for name, camera in self.human_render_cameras.items():
                    if camera_name is not None and name != camera_name:
                        continue
                    assert (
                        camera.config.shader_config.shader_pack  # pyright: ignore[reportOptionalMemberAccess]
                        not in [
                            "rt",
                            "rt-fast",
                            "rt-med",
                        ]
                    ), "ray tracing shaders do not work with parallel rendering"
                    camera.capture()
                    rgb = camera.get_obs(
                        rgb=True, depth=False, segmentation=False, position=False
                    )["rgb"]
                    image_data[name] = rgb
        else:
            for name, camera in self.human_render_cameras.items():
                if camera_name is not None and name != camera_name:
                    continue
                camera.capture()
                rgb = camera.get_obs(
                    rgb=True, depth=False, segmentation=False, position=False
                )["rgb"]
                image_data[name] = rgb
        return image_data
