from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import sapien
import sapien.physx as physx
import torch
import viser
import time

from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.types import Array


@dataclass
class ViserEntityDisplay:
    visual_mesh_handles: list[viser.SceneNodeHandle]
    collision_mesh_handles: list[viser.SceneNodeHandle]
    transparent_visual_mesh_handles: list[viser.SceneNodeHandle]
    transparent_collision_mesh_handles: list[viser.SceneNodeHandle]
    coordinate_frame_handle: viser.FrameHandle
    show_collision_checkbox: Any
    show_coordinate_frame_checkbox: Any
    is_transparent: bool = False


class ViserVisualizer:
    """
    Visualizer that mirrors a ManiSkillScene in viser (https://viser.studio/) alongside the
    default SAPIEN renderer. ManiSkillScene always runs its Sapien path, then optionally calls
    the matching methods here when ManiSkillScene.viser_visualizer is set
    (i.e. visualizer_backend == "viser").
    """

    def __init__(self, scene: "ManiSkillScene", server: Optional[Any] = None):
        assert (
            not scene.gpu_sim_enabled
        ), "ViserVisualizer currently only supports the CPU simulation backend"
        assert (
            len(scene.sub_scenes) == 1
        ), "ViserVisualizer currently only supports a single environment (num_envs == 1)"
        self.scene = scene
        # ManiSkillScene._viser_server is reused across reconfigures (each of which recreates
        # this ViserVisualizer) so we do not spawn a new server and orphan the browser tab.
        if server is None:
            server = viser.ViserServer()
        else:
            server.scene.reset()
            server.gui.reset()
        self.server = server
        self.server.scene.world_axes.visible = True
        self.server.scene.world_axes.scale = 0.25
        self.server.scene.add_grid(
            "/xy_grid",
            width=2.0,
            height=2.0,
            plane="xy",
            cell_size=0.05,
            section_size=0.25,
            cell_color=(220, 220, 220),
            section_color=(200, 200, 200),
            plane_opacity=0.05,
        )
        self.paused = False
        self.pause_checkbox = self.server.gui.add_checkbox(
            "Pause simulation", initial_value=False
        )
        self.pause_checkbox.on_update(
            lambda _: setattr(self, "paused", self.pause_checkbox.value)
        )
        self.entity_displays: dict[str, ViserEntityDisplay] = {}
        self.articulation_link_frames: dict[str, dict[str, viser.FrameHandle]] = {}
        """maps articulation name (as registered in scene.articulations) to a map of link name -> the viser frame its visual shapes are parented under"""
        self.actor_frames: dict[str, viser.FrameHandle] = {}
        """maps actor name (as registered in scene.actors) to the viser frame its visual shapes are parented under"""
        self._debug_viz_counter = 0
        """monotonically increasing counter used to name ad-hoc debug visualization nodes"""

    def load_articulation(self, name: str, articulation: Articulation) -> None:
        """Displays an articulation in the viser scene, reading each link's visual/collision geometry and
        pose directly off the live Articulation struct (no separate builder bookkeeping needed)."""
        link_frames: dict[str, viser.FrameHandle] = {}
        for link in articulation.links:
            node_prefix = f"/{name}/{link.name}"
            pose = common.to_numpy(link.pose.raw_pose)[0]
            frame = self.server.scene.add_frame(
                node_prefix,
                show_axes=False,
                position=tuple(pose[:3]),
                wxyz=tuple(pose[3:7]),
            )
            link_frames[link.name] = frame
            display = self._add_entity_display(f"{name}/{link.name}", node_prefix)
            self._add_meshes(display, node_prefix, link._objs[0].entity, link._objs[0])
        self.articulation_link_frames[name] = link_frames

    def load_actor(self, name: str, actor: Actor) -> None:
        """Displays an actor in the viser scene, reading its visual/collision geometry and pose directly off
        the live Actor struct (no separate builder bookkeeping needed)."""
        pose = common.to_numpy(actor.pose.raw_pose)[0]
        frame = self.server.scene.add_frame(
            f"/{name}",
            show_axes=False,
            position=tuple(pose[:3]),
            wxyz=tuple(pose[3:7]),
        )
        self.actor_frames[name] = frame
        entity = actor._objs[0]
        component = next(
            (c for c in entity.components if isinstance(c, physx.PhysxRigidBaseComponent)),
            None,
        )
        display = self._add_entity_display(name, f"/{name}")
        self._add_meshes(display, f"/{name}", entity, component)

    def _add_entity_display(
        self, label: str, node_prefix: str
    ) -> ViserEntityDisplay:
        coordinate_frame = self.server.scene.add_frame(
            f"{node_prefix}/coordinate_frame",
            show_axes=True,
            axes_length=0.25,
            axes_radius=0.005,
            position=(0.0, 0.0, 0.0),
            wxyz=(1.0, 0.0, 0.0, 0.0),
            visible=False,
        )
        with self.server.gui.add_folder(label, expand_by_default=True):
            show_collision_checkbox = self.server.gui.add_checkbox(
                "Collision mesh", initial_value=False
            )
            show_coordinate_frame_checkbox = self.server.gui.add_checkbox(
                "Coordinate frame", initial_value=False
            )

        display = ViserEntityDisplay(
            visual_mesh_handles=[],
            collision_mesh_handles=[],
            transparent_visual_mesh_handles=[],
            transparent_collision_mesh_handles=[],
            coordinate_frame_handle=coordinate_frame,
            show_collision_checkbox=show_collision_checkbox,
            show_coordinate_frame_checkbox=show_coordinate_frame_checkbox,
        )
        self.entity_displays[node_prefix] = display
        coordinate_frame.on_click(lambda _: self._toggle_entity_transparency(display))
        show_collision_checkbox.on_update(
            lambda _: self._update_entity_mesh_visibility(display)
        )
        show_coordinate_frame_checkbox.on_update(
            lambda _: self._update_entity_frame_visibility(display)
        )
        return display

    def _add_meshes(
        self,
        display: ViserEntityDisplay,
        node_prefix: str,
        entity: sapien.Entity,
        component: Optional[physx.PhysxRigidBaseComponent],
    ) -> None:
        """Adds the (local-frame) visual and collision meshes of a Sapien entity to the viser scene, if
        any (entities with only a plane collision/visual shape, e.g. the ground, have none). Only one of the
        two is shown at a time when collision meshes are available, toggled by this entity's collision
        checkbox.

        Visual meshes are added one per render shape/part (rather than merged into one mesh) so that each
        keeps its own material/texture (e.g. a table's wood texture) instead of being flattened to a single
        color."""
        from mani_skill.utils.geometry.trimesh_utils import (
            get_actor_visual_meshes,
            get_component_meshes,
            merge_meshes,
        )

        show_collision = display.show_collision_checkbox.value
        for i, visual_mesh in enumerate(get_actor_visual_meshes(entity)):
            handle, transparent_handle = self._add_clickable_mesh_pair(
                display,
                f"{node_prefix}/visual/{i}",
                f"{node_prefix}/transparent_visual/{i}",
                visual_mesh,
            )
            handle.visible = not show_collision
            transparent_handle.visible = False
            display.visual_mesh_handles.append(handle)
            display.transparent_visual_mesh_handles.append(transparent_handle)
        collision_mesh = (
            merge_meshes(get_component_meshes(component))
            if component is not None
            else None
        )
        if collision_mesh is not None:
            handle, transparent_handle = self._add_clickable_mesh_pair(
                display,
                f"{node_prefix}/collision",
                f"{node_prefix}/transparent_collision",
                collision_mesh,
            )
            handle.visible = show_collision
            transparent_handle.visible = False
            display.collision_mesh_handles.append(handle)
            display.transparent_collision_mesh_handles.append(transparent_handle)

    def _add_clickable_mesh_pair(
        self,
        display: ViserEntityDisplay,
        opaque_name: str,
        transparent_name: str,
        mesh,
    ) -> tuple[viser.SceneNodeHandle, viser.SceneNodeHandle]:
        opaque_handle = self.server.scene.add_mesh_trimesh(opaque_name, mesh)
        transparent_handle = self.server.scene.add_mesh_trimesh(
            transparent_name,
            self._make_transparent_mesh(mesh, alpha=0.5),
            visible=False,
            cast_shadow=False,
        )
        opaque_handle.on_click(lambda _: self._toggle_entity_transparency(display))
        transparent_handle.on_click(lambda _: self._toggle_entity_transparency(display))
        return opaque_handle, transparent_handle

    def _make_transparent_mesh(self, mesh, alpha: float):
        transparent_mesh = mesh.copy()
        alpha_uint8 = int(round(alpha * 255))
        visual = transparent_mesh.visual

        def with_alpha(colors):
            colors = np.array(colors, copy=True)
            uses_float_alpha = np.issubdtype(colors.dtype, np.floating)
            opaque_alpha = 1.0 if uses_float_alpha else 255
            transparent_alpha = alpha if uses_float_alpha else alpha_uint8
            if colors.shape[-1] == 3:
                colors = np.concatenate(
                    [
                        colors,
                        np.full(
                            (*colors.shape[:-1], 1),
                            opaque_alpha,
                            dtype=colors.dtype,
                        ),
                    ],
                    axis=-1,
                )
            colors[..., 3] = transparent_alpha
            return colors

        if hasattr(visual, "face_colors") and len(visual.face_colors) > 0:
            visual.face_colors = with_alpha(visual.face_colors)
        if hasattr(visual, "vertex_colors") and len(visual.vertex_colors) > 0:
            visual.vertex_colors = with_alpha(visual.vertex_colors)

        material = getattr(visual, "material", None)
        if material is not None:
            material = material.copy()
            main_color = with_alpha(material.main_color)
            if hasattr(material, "diffuse"):
                material.diffuse = main_color
            if hasattr(material, "baseColorFactor"):
                material.baseColorFactor = main_color
                material.alphaMode = "BLEND"
            visual.material = material

        return transparent_mesh

    def _update_entity_mesh_visibility(self, display: ViserEntityDisplay) -> None:
        show_collision = (
            display.show_collision_checkbox.value
            and len(display.collision_mesh_handles) > 0
        )
        for handle in display.visual_mesh_handles:
            handle.visible = not show_collision and not display.is_transparent
        for handle in display.transparent_visual_mesh_handles:
            handle.visible = not show_collision and display.is_transparent
        for handle in display.collision_mesh_handles:
            handle.visible = show_collision and not display.is_transparent
        for handle in display.transparent_collision_mesh_handles:
            handle.visible = show_collision and display.is_transparent

    def _toggle_entity_transparency(self, display: ViserEntityDisplay) -> None:
        display.is_transparent = not display.is_transparent
        self._update_entity_mesh_visibility(display)

    def _update_entity_frame_visibility(self, display: ViserEntityDisplay) -> None:
        display.coordinate_frame_handle.visible = (
            display.show_coordinate_frame_checkbox.value
        )

    def sync_articulation_poses(self) -> None:
        """Synchronizes the poses of all displayed articulation links with their live simulation state, read
        directly off each Link's (already forward-kinematics-computed) pose."""
        for name, link_frames in self.articulation_link_frames.items():
            articulation = self.scene.articulations.get(name)
            if articulation is None:
                continue
            for link in articulation.links:
                frame = link_frames.get(link.name)
                if frame is None:
                    continue
                pose = common.to_numpy(link.pose.raw_pose)[0]
                frame.position = pose[:3]
                frame.wxyz = pose[3:7]

    def sync_actor_poses(self) -> None:
        """Synchronizes the poses of all displayed actors with their live simulation state."""
        for name, frame in self.actor_frames.items():
            actor = self.scene.actors.get(name)
            if actor is None:
                continue
            pose = common.to_numpy(actor.pose.raw_pose)[0]
            frame.position = pose[:3]
            frame.wxyz = pose[3:7]

    def sync(self) -> None:
        """Synchronizes all displayed articulations and actors with their live simulation state."""
        self.sync_articulation_poses()
        self.sync_actor_poses()

    def wait_while_paused(self) -> None:
        """Blocks simulation stepping while the Viser pause control is active."""
        while self.paused:
            self.sync()
            time.sleep(0.01)

    def add_camera(
        self,
        name,
        pose,
        width,
        height,
        near,
        far,
        fovy: Union[float, list, None] = None,
        intrinsic: Union[Array, None] = None,
        mount: Union[Actor, Link, None] = None,
    ) -> None:
        """No-op: Sapien cameras are created by ManiSkillScene.add_camera before this is called."""
        pass

    def update_render(
        self, update_sensors: bool = True, update_human_render_cameras: bool = True
    ):
        self.sync()

    def add_point_light(
        self,
        position,
        color,
        shadow=False,
        shadow_near=0.1,
        shadow_far=10.0,
        shadow_map_size=2048,
        scene_idxs: Optional[list[int]] = None,
    ):
        pass

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
        scene_idxs: Optional[list[int]] = None,
    ):
        pass

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
        scene_idxs: Optional[list[int]] = None,
    ):
        pass

    def add_area_light_for_ray_tracing(
        self,
        pose: sapien.Pose,
        color,
        half_width: float,
        half_height: float,
        scene_idxs=None,
    ):
        pass

    # -------------------------------------------------------------------------- #
    # Debug visualization helpers (mirrors Sapien viewer debug draw APIs)
    # -------------------------------------------------------------------------- #

    def _next_debug_name(self, kind: str) -> str:
        self._debug_viz_counter += 1
        return f"/debug/{kind}/{self._debug_viz_counter}"

    @staticmethod
    def _to_rgb255(color) -> np.ndarray:
        """Convert a float [0, 1] or uint8 RGB(A) color to uint8 RGB."""
        color = np.asarray(color, dtype=np.float64).reshape(-1)
        rgb = color[:3]
        if rgb.max() <= 1.0:
            rgb = rgb * 255.0
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def add_bounding_box(
        self,
        pose: sapien.Pose,
        half_size: np.ndarray,
        color: np.ndarray,
        line_width: float = 1.0,
    ):
        """Add a wireframe axis-aligned box in the box's local frame, transformed by ``pose``."""
        corners = np.array(
            [
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, -1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
                [-1, -1, 1],
            ],
            dtype=np.float32,
        )
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        points = np.stack([corners[list(edge)] for edge in edges], axis=0)
        half_size = np.asarray(half_size, dtype=np.float32).reshape(3)
        return self.server.scene.add_line_segments(
            self._next_debug_name("bbox"),
            points=points,
            colors=self._to_rgb255(color),
            line_width=line_width,
            scale=tuple(half_size),
            position=tuple(np.asarray(pose.p, dtype=np.float32).reshape(3)),
            wxyz=tuple(np.asarray(pose.q, dtype=np.float32).reshape(4)),
        )

    def update_bounding_box(self, box, pose: sapien.Pose, half_size: np.ndarray):
        half_size = np.asarray(half_size, dtype=np.float32).reshape(3)
        box.position = tuple(np.asarray(pose.p, dtype=np.float32).reshape(3))
        box.wxyz = tuple(np.asarray(pose.q, dtype=np.float32).reshape(4))
        box.scale = tuple(half_size)

    def remove_bounding_box(self, box):
        box.remove()

    def draw_aabb(self, lower, upper, color, line_width: float = 1.0):
        lower = np.asarray(lower, dtype=np.float32).reshape(3)
        upper = np.asarray(upper, dtype=np.float32).reshape(3)
        pose = sapien.Pose((lower + upper) / 2)
        half_size = (upper - lower) / 2
        return self.add_bounding_box(pose, half_size, color, line_width)

    def update_aabb(self, aabb, lower, upper):
        lower = np.asarray(lower, dtype=np.float32).reshape(3)
        upper = np.asarray(upper, dtype=np.float32).reshape(3)
        pose = sapien.Pose((lower + upper) / 2)
        half_size = (upper - lower) / 2
        self.update_bounding_box(aabb, pose, half_size)

    def remove_aabb(self, aabb):
        self.remove_bounding_box(aabb)

    def add_3d_point_list(
        self, points: np.ndarray, color: Optional[np.ndarray] = None
    ):
        """
        Add a 3D point list to the visualizer.

        Args:
            points: Array of 3D points, shape [N, 3].
            color: Optional color. None defaults to white. Can be:
                - shape [3] or [4]: single RGB(A) color for all points
                - shape [N, 3] or [N, 4]: per-point RGB(A) colors
        """
        points = np.asarray(points, dtype=np.float32)
        assert points.ndim == 2 and points.shape[1] == 3, "points must be shape [N, 3]"
        n_points = points.shape[0]

        if color is None:
            colors = np.full((n_points, 3), 255, dtype=np.uint8)
        else:
            color = np.asarray(color, dtype=np.float64)
            if color.ndim == 1:
                colors = np.tile(self._to_rgb255(color).reshape(1, 3), (n_points, 1))
            elif color.ndim == 2:
                assert (
                    color.shape[0] == n_points
                ), "color must have same number of points as points"
                assert color.shape[1] in (
                    3,
                    4,
                ), "color must be shape [N, 3] or [N, 4]"
                if color.max() <= 1.0:
                    color = color * 255.0
                colors = np.clip(color[:, :3], 0, 255).astype(np.uint8)
            else:
                raise ValueError("color must be 1D or 2D array")

        return self.server.scene.add_point_cloud(
            self._next_debug_name("points"),
            points=points,
            colors=colors,
            point_size=0.003,
            point_shape="circle",
            point_shading="gradient",
        )

    def remove_3d_point_list(self, pointset_obj):
        """Remove a 3D point list from the visualizer."""
        pointset_obj.remove()

    def add_coordinate_frame(
        self, pose: sapien.Pose, length: float = 0.1, radius: float = 0.02
    ):
        return self.server.scene.add_frame(
            self._next_debug_name("frame"),
            show_axes=True,
            axes_length=length,
            axes_radius=radius,
            position=tuple(np.asarray(pose.p, dtype=np.float32).reshape(3)),
            wxyz=tuple(np.asarray(pose.q, dtype=np.float32).reshape(4)),
        )

    def remove_coordinate_frame(self, node):
        """Remove a coordinate frame from the visualizer."""
        node.remove()

