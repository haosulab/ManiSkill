from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import sapien
import sapien.physx as physx
import sapien.render
import torch
from sapien.render import RenderCameraComponent

from mani_skill.sensors.base_sensor import BaseSensor
from mani_skill.sensors.camera import Camera
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.render_camera import RenderCamera
from mani_skill.utils.structs.types import Array, Device, SimConfig


class ManiSkillScene:
    """
    Class that manages a list of sub-scenes (sapien.Scene). In CPU simulation there should only be one sub-scene.
    In GPU simulation, there can be many sub-scenes, and this wrapper ensures that use calls to many of the original sapien.Scene API
    are applied to all sub-scenes. This includes calls to change object poses, velocities, drive targets etc.

    This wrapper also helps manage GPU states if GPU simulation is used
    """

    def __init__(
        self,
        sub_scenes: List[sapien.Scene],
        sim_cfg: SimConfig,
        debug_mode: bool = True,
        device: Device = None,
    ):
        self.sub_scenes = sub_scenes
        self.px: Union[physx.PhysxCpuSystem, physx.PhysxGpuSystem] = self.sub_scenes[
            0
        ].physx_system
        self.sim_cfg = sim_cfg
        self._gpu_sim_initialized = False
        self.debug_mode = debug_mode
        self.device = device

        self.render_system_group: sapien.render.RenderSystemGroup = None
        self.camera_groups: Dict[str, sapien.render.RenderCameraGroup] = OrderedDict()

        self.actors: Dict[str, Actor] = OrderedDict()
        self.articulations: Dict[str, Articulation] = OrderedDict()

        self.sensors: Dict[str, BaseSensor] = OrderedDict()
        self.human_render_cameras: Dict[str, Camera] = OrderedDict()

        self._reset_mask = torch.ones(len(sub_scenes), dtype=bool, device=self.device)
        """Used internally by various objects like Actor, Link, and Controllers to auto mask out sub-scenes so they do not get modified during
        partial env resets"""

    @property
    def timestep(self):
        return self.px.timestep

    @timestep.setter
    def timestep(self, timestep):
        self.px.timestep = timestep

    def set_timestep(self, timestep):
        self.timestep = timestep

    def get_timestep(self):
        return self.timestep

    def create_actor_builder(self):
        from ..utils.building.actor_builder import ActorBuilder

        return ActorBuilder().set_scene(self)

    def create_articulation_builder(self):
        from ..utils.building.articulation_builder import ArticulationBuilder

        return ArticulationBuilder().set_scene(self)

    def create_urdf_loader(self):
        from ..utils.building.urdf_loader import URDFLoader

        loader = URDFLoader()
        loader.set_scene(self)
        return loader

    def create_physical_material(
        self, static_friction: float, dynamic_friction: float, restitution: float
    ):
        return sapien.physx.PhysxMaterial(
            static_friction, dynamic_friction, restitution
        )

    def remove_actor(self, actor):
        if physx.is_gpu_enabled():
            raise NotImplementedError(
                "Cannot remove actors after creating them in GPU sim at the moment"
            )
        else:
            self.sub_scenes[0].remove_entity(actor)

    def remove_articulation(self, articulation: Articulation):
        if physx.is_gpu_enabled():
            raise NotImplementedError(
                "Cannot remove articulations after creating them in GPU sim at the moment"
            )
        else:
            entities = [l.entity for l in articulation._objs[0].links]
            for e in entities:
                self.sub_scenes[0].remove_entity(e)

    def add_camera(
        self,
        name,
        pose: Pose,
        width: int,
        height: int,
        fovy: float,
        near: float,
        far: float,
    ) -> RenderCamera:
        cameras = []
        pose = Pose.create(pose)
        for i, scene in enumerate(self.sub_scenes):
            camera_mount = sapien.Entity()
            camera = RenderCameraComponent(width, height)
            if isinstance(fovy, float) or isinstance(fovy, int):
                camera.set_fovy(fovy, compute_x=True)
            else:
                camera.set_fovy(fovy[i], compute_x=True)
            if isinstance(near, float) or isinstance(near, int):
                camera.near = near
            else:
                camera.near = near[i]
            if isinstance(far, float) or isinstance(far, int):
                camera.far = far
            else:
                camera.far = far[i]
            camera_mount.add_component(camera)
            if len(pose) == 1:
                camera.local_pose = pose.sp
            else:
                camera.local_pose = pose[i].sp

            scene.add_entity(camera_mount)
            camera_mount.name = f"scene-{i}_{name}"
            camera.name = f"scene-{i}_{name}"
            cameras.append(camera)
        return RenderCamera.create(cameras, self)

    def add_mounted_camera(
        self,
        name,
        mount: Union[Actor, Link],
        pose: Pose,
        width,
        height,
        fovy,
        near,
        far,
    ) -> RenderCamera:
        cameras = []
        pose = Pose.create(pose)
        for i, scene in enumerate(self.sub_scenes):
            camera = RenderCameraComponent(width, height)
            if isinstance(fovy, float) or isinstance(fovy, int):
                camera.set_fovy(fovy, compute_x=True)
            else:
                camera.set_fovy(fovy[i], compute_x=True)
            if isinstance(near, float) or isinstance(near, int):
                camera.near = near
            else:
                camera.near = near[i]
            if isinstance(far, float) or isinstance(far, int):
                camera.far = far
            else:
                camera.far = far[i]
            if physx.is_gpu_enabled():
                if isinstance(mount, Actor):
                    camera.set_gpu_pose_batch_index(
                        mount._objs[i]
                        .find_component_by_type(physx.PhysxRigidBodyComponent)
                        .gpu_pose_index
                    )
                elif isinstance(mount, Link):
                    camera.set_gpu_pose_batch_index(mount._objs[i].gpu_pose_index)
                else:
                    raise ValueError(
                        f"Tried to mount camera on object of type {mount.__class__}"
                    )

            if isinstance(mount, Link):
                mount._objs[i].entity.add_component(camera)
            else:
                mount._objs[i].add_component(camera)
            if len(pose) == 1:
                camera.local_pose = pose.sp
            else:
                camera.local_pose = pose[i].sp
            camera.name = f"scene-{i}_{name}"
            cameras.append(camera)
        return RenderCamera.create(cameras, self, mount=mount)

    # def remove_camera(self, camera):
    #     self.remove_entity(camera.entity)

    # def get_cameras(self):
    #     return self.render_system.cameras

    # def get_mounted_cameras(self):
    #     return self.get_cameras()

    def step(self):
        self.px.step()

    def update_render(self):
        if physx.is_gpu_enabled():
            if self.render_system_group is None:
                self._setup_gpu_rendering()
                self._gpu_setup_sensors(self.sensors)
                self._gpu_setup_sensors(self.human_render_cameras)
            self.render_system_group.update_render()
        else:
            self.sub_scenes[0].update_render()

    def get_contacts(self):
        return self.px.get_contacts()

    def get_all_actors(self):
        """
        Returns list of all sapien.Entity objects that have rigid dynamic and static components across all sub scenes
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
        body0: Optional[Union[sapien.Entity, sapien.physx.PhysxRigidBaseComponent]],
        pose0: sapien.Pose,
        body1: Union[sapien.Entity, sapien.physx.PhysxRigidBaseComponent],
        pose1: sapien.Pose,
    ):
        if body0 is None:
            c0 = None
        elif isinstance(body0, sapien.Entity):
            c0 = next(
                c
                for c in body0.components
                if isinstance(c, sapien.physx.PhysxRigidBaseComponent)
            )
        else:
            c0 = body0

        assert body1 is not None
        if isinstance(body1, sapien.Entity):
            e1 = body1
            c1 = next(
                c
                for c in body1.components
                if isinstance(c, sapien.physx.PhysxRigidBaseComponent)
            )
        else:
            e1 = body1.entity
            c1 = body1

        drive = sapien.physx.PhysxDriveComponent(c1)
        drive.parent = c0
        drive.pose_in_child = pose1
        drive.pose_in_parent = pose0
        e1.add_component(drive)
        return drive

    def create_connection(
        self,
        body0: Optional[Union[sapien.Entity, sapien.physx.PhysxRigidBaseComponent]],
        pose0: sapien.Pose,
        body1: Union[sapien.Entity, sapien.physx.PhysxRigidBaseComponent],
        pose1: sapien.Pose,
    ):
        if body0 is None:
            c0 = None
        elif isinstance(body0, sapien.Entity):
            c0 = next(
                c
                for c in body0.components
                if isinstance(c, sapien.physx.PhysxRigidBaseComponent)
            )
        else:
            c0 = body0

        assert body1 is not None
        if isinstance(body1, sapien.Entity):
            e1 = body1
            c1 = next(
                c
                for c in body1.components
                if isinstance(c, sapien.physx.PhysxRigidBaseComponent)
            )
        else:
            e1 = body1.entity
            c1 = body1

        connection = sapien.physx.PhysxDistanceJointComponent(c1)
        connection.parent = c0
        connection.pose_in_child = pose1
        connection.pose_in_parent = pose0
        e1.add_component(connection)
        connection.set_limit(0, 0)
        return connection

    def create_gear(
        self,
        body0: Optional[Union[sapien.Entity, sapien.physx.PhysxRigidBaseComponent]],
        pose0: sapien.Pose,
        body1: Union[sapien.Entity, sapien.physx.PhysxRigidBaseComponent],
        pose1: sapien.Pose,
    ):
        if body0 is None:
            c0 = None
        elif isinstance(body0, sapien.Entity):
            c0 = next(
                c
                for c in body0.components
                if isinstance(c, sapien.physx.PhysxRigidBaseComponent)
            )
        else:
            c0 = body0

        assert body1 is not None
        if isinstance(body1, sapien.Entity):
            e1 = body1
            c1 = next(
                c
                for c in body1.components
                if isinstance(c, sapien.physx.PhysxRigidBaseComponent)
            )
        else:
            e1 = body1.entity
            c1 = body1

        gear = sapien.physx.PhysxGearComponent(c1)
        gear.parent = c0
        gear.pose_in_child = pose1
        gear.pose_in_parent = pose0
        e1.add_component(gear)
        return gear

    # @property
    # def render_id_to_visual_name(self):
    #     # TODO
    #     return

    @property
    def ambient_light(self):
        return self.sub_scenes[0].ambient_light

    @ambient_light.setter
    def ambient_light(self, color):
        for scene in self.sub_scenes:
            scene.render_system.ambient_light = color

    def set_ambient_light(self, color):
        self.ambient_light = color

    def add_point_light(
        self,
        position,
        color,
        shadow=False,
        shadow_near=0.1,
        shadow_far=10.0,
        shadow_map_size=2048,
    ):
        for scene in self.sub_scenes:
            entity = sapien.Entity()
            light = sapien.render.RenderPointLightComponent()
            entity.add_component(light)
            light.color = color
            light.shadow = shadow
            light.shadow_near = shadow_near
            light.shadow_far = shadow_far
            light.shadow_map_size = shadow_map_size
            light.pose = sapien.Pose(position)
            scene.add_entity(entity)
        return light

    def add_directional_light(
        self,
        direction,
        color,
        shadow=False,
        position=[0, 0, 0],
        shadow_scale=10.0,
        shadow_near=-10.0,
        shadow_far=10.0,
        shadow_map_size=2048,
    ):
        for scene in self.sub_scenes:
            entity = sapien.Entity()
            light = sapien.render.RenderDirectionalLightComponent()
            entity.add_component(light)
            light.color = color
            light.shadow = shadow
            light.shadow_near = shadow_near
            light.shadow_far = shadow_far
            light.shadow_half_size = shadow_scale
            light.shadow_map_size = shadow_map_size
            light.pose = sapien.Pose(
                position, sapien.math.shortest_rotation([1, 0, 0], direction)
            )
            scene.add_entity(entity)
        return

    def add_spot_light(
        self,
        position,
        direction,
        inner_fov: float,
        outer_fov: float,
        color,
        shadow=False,
        shadow_near=0.1,
        shadow_far=10.0,
        shadow_map_size=2048,
    ):
        for scene in self.sub_scenes:
            entity = sapien.Entity()
            light = sapien.render.RenderSpotLightComponent()
            entity.add_component(light)
            light.color = color
            light.shadow = shadow
            light.shadow_near = shadow_near
            light.shadow_far = shadow_far
            light.shadow_map_size = shadow_map_size
            light.inner_fov = inner_fov
            light.outer_fov = outer_fov
            light.pose = sapien.Pose(
                position, sapien.math.shortest_rotation([1, 0, 0], direction)
            )
            scene.add_entity(entity)
        return

    def add_area_light_for_ray_tracing(
        self, pose: sapien.Pose, color, half_width: float, half_height: float
    ):
        for scene in self.sub_scenes:
            entity = sapien.Entity()
            light = sapien.render.RenderParallelogramLightComponent()
            entity.add_component(light)
            light.set_shape(half_width, half_height)
            light.color = color
            light.pose = pose
            scene.add_entity(entity)
        return

    # def remove_light(self, light):
    #     self.remove_entity(light.entity)

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
    # Additional useful properties
    # ---------------------------------------------------------------------------- #
    @property
    def num_envs(self):
        return len(self.sub_scenes)

    # -------------------------------------------------------------------------- #
    # Simulation state (required for MPC)
    # -------------------------------------------------------------------------- #
    def get_sim_state(self) -> torch.Tensor:
        """Get simulation state. Returns a dictionary with two nested dictionaries "actors" and "articulations".
        In the nested dictionaries they map the actor/articulation name to a vector of shape (N, D) for N parallel
        environments and D dimensions of padded state per environment.

        Note that static actor data are not included. It is expected that an environment reconstructs itself in a deterministic manner such that
        the same static actors always have the same states"""
        state_dict = dict()
        state_dict["actors"] = dict()
        state_dict["articulations"] = dict()
        for actor in self.actors.values():
            if actor.px_body_type == "static":
                continue
            state_dict["actors"][actor.name] = actor.get_state().clone()
        for articulation in self.articulations.values():
            state_dict["articulations"][
                articulation.name
            ] = articulation.get_state().clone()
        return state_dict

    def set_sim_state(self, state: Dict):
        for actor_id, actor_state in state["actors"].items():
            if len(actor_state.shape) == 1:
                actor_state = actor_state[None, :]
            self.actors[actor_id].set_state(actor_state)
        for art_id, art_state in state["articulations"].items():
            if len(art_state.shape) == 1:
                art_state = art_state[None, :]
            self.articulations[art_id].set_state(art_state)

    # ---------------------------------------------------------------------------- #
    # GPU Simulation Management
    # ---------------------------------------------------------------------------- #
    def _setup_gpu(self):
        """
        Start the GPU simulation and allocate all buffers and initialize objects
        """
        self.px.gpu_init()
        self.non_static_actors: List[Actor] = []
        # find non static actors, and set data indices that are now available after gpu_init was called
        for actor in self.actors.values():
            if actor.px_body_type == "static":
                continue
            self.non_static_actors.append(actor)
            actor._body_data_index  # only need to access this attribute to populate it

        for articulation in self.articulations.values():
            articulation._data_index
            for link in articulation.links:
                link._body_data_index

        # As physx_system.gpu_init() was called a single physx step was also taken. So we need to reset
        # all the actors and articulations to their original poses as they likely have collided
        for actor in self.non_static_actors:
            actor.set_pose(actor._builder_initial_pose)
        self.px.cuda_rigid_body_data.torch()[:, 7:] = (
            self.px.cuda_rigid_body_data.torch()[:, 7:] * 0
        )  # zero out all velocities
        self.px.gpu_apply_rigid_dynamic_data()
        self.px.gpu_apply_articulation_root_velocity()
        self.px.cuda_articulation_qvel.torch()[:, :] = (
            self.px.cuda_articulation_qvel.torch() * 0
        )  # zero out all q velocities
        self.px.gpu_apply_articulation_qvel()

        self._gpu_sim_initialized = True
        self._gpu_fetch_all()

    def _gpu_apply_all(self):
        """
        Calls gpu_apply to update all body data, qpos, qvel, qf, and root poses
        """
        self.px.gpu_apply_rigid_dynamic_data()
        self.px.gpu_apply_articulation_qpos()
        self.px.gpu_apply_articulation_qvel()
        self.px.gpu_apply_articulation_qf()
        self.px.gpu_apply_articulation_root_pose()
        self.px.gpu_apply_articulation_root_velocity()
        self.px.gpu_apply_articulation_target_position()
        self.px.gpu_apply_articulation_target_velocity()

    def _gpu_fetch_all(self):
        """
        Queries simulation for all relevant GPU data. Note that this has some overhead.
        Should only be called at most once per simulation step as this automatically queries all data for all
        objects built in the scene.
        """
        if len(self.non_static_actors) > 0:
            self.px.gpu_fetch_rigid_dynamic_data()

        if len(self.articulations) > 0:
            self.px.gpu_fetch_articulation_link_pose()
            self.px.gpu_fetch_articulation_link_velocity()
            self.px.gpu_fetch_articulation_qpos()
            self.px.gpu_fetch_articulation_qvel()
            self.px.gpu_fetch_articulation_target_qpos()
            self.px.gpu_fetch_articulation_target_qvel()

            # unused fetches
            # self.px.gpu_fetch_articulation_qacc()

    # ---------------------------------------------------------------------------- #
    # CPU/GPU sim Rendering Code
    # ---------------------------------------------------------------------------- #
    def _get_all_render_bodies(
        self,
    ) -> List[Tuple[sapien.render.RenderBodyComponent, int]]:
        all_render_bodies = []
        for actor in self.actors.values():
            if actor.px_body_type == "static":
                continue
            all_render_bodies += [
                (
                    entity.find_component_by_type(sapien.render.RenderBodyComponent),
                    entity.find_component_by_type(
                        physx.PhysxRigidDynamicComponent
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
        """
        Prepares the scene for GPU parallelized rendering to enable taking e.g. RGB images
        """
        for rb, gpu_pose_index in self._get_all_render_bodies():
            if rb is not None:
                for s in rb.render_shapes:
                    s.set_gpu_pose_batch_index(gpu_pose_index)
        self.render_system_group = sapien.render.RenderSystemGroup(
            [s.render_system for s in self.sub_scenes]
        )
        self.render_system_group.set_cuda_poses(self.px.cuda_rigid_body_data)

    def _gpu_setup_sensors(self, sensors: Dict[str, BaseSensor]):
        for name, sensor in sensors.items():
            if isinstance(sensor, Camera):
                camera_group = self.render_system_group.create_camera_group(
                    sensor.camera._render_cameras,
                    sensor.texture_names,
                )
                sensor.camera.camera_group = camera_group
                self.camera_groups[name] = camera_group
            else:
                raise NotImplementedError(
                    f"This sensor {sensor} has not been implemented yet on the GPU"
                )
