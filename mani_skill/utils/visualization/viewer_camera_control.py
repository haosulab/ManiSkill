from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import sapien
from sapien import internal_renderer as R
from sapien.utils.viewer.plugin import Plugin, copy_to_clipboard
from transforms3d.quaternions import mat2quat

from mani_skill.sensors.camera import Camera
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.pose import Pose

if TYPE_CHECKING:
    from mani_skill.envs.sapien_env import BaseEnv


_VIEWER_GL_POSE = sapien.Pose([0, 0, 0], [-0.5, -0.5, 0.5, 0.5])
_CAMERA_LINESET_VERTICES = [
    0,
    0,
    0,
    1,
    1,
    -1,
    0,
    0,
    0,
    -1,
    1,
    -1,
    0,
    0,
    0,
    1,
    -1,
    -1,
    0,
    0,
    0,
    -1,
    -1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    -1,
    -1,
    1,
    -1,
    1,
    1,
    -1,
    1,
    1.2,
    -1,
    0,
    2,
    -1,
    0,
    2,
    -1,
    -1,
    1.2,
    -1,
    -1,
    1.2,
    -1,
    1,
    1.2,
    -1,
]
_INACTIVE_CAMERA_LINESET_COLORS = [0.9254901961, 0.5764705882, 0.1882352941, 1.0] * 22
_ACTIVE_CAMERA_LINESET_COLORS = [0.1882352941, 0.7921568627, 0.3882352941, 1.0] * 22


class ViewerCameraControlPlugin(Plugin):
    """Edit ManiSkill cameras directly from the SAPIEN viewer."""

    def __init__(
        self,
        env: "BaseEnv",
        enabled: bool = True,
        selection_radius_px: float = 18.0,
    ):
        self.env = env
        self.enabled = enabled
        self.selection_radius_px = selection_radius_px
        self._active_camera_name: Optional[str] = None
        self._line_sets = []
        self._line_set_names: list[str] = []
        self._line_set_active_name: Optional[str] = None
        self._line_set_model = None
        self._active_line_set_model = None
        self._ui_window = None
        self._gizmo = None
        self._target_gizmo = None
        self._camera_selector = None
        self._camera_targets: dict[str, np.ndarray] = {}
        self._camera_ups: dict[str, np.ndarray] = {}
        self._initial_local_poses: dict[str, sapien.Pose] = {}
        self._initial_targets: dict[str, np.ndarray] = {}
        self._initial_ups: dict[str, np.ndarray] = {}
        self.default_target_distance = 0.25

    def init(self, viewer):
        super().init(viewer)
        if hasattr(self.viewer, "register_click_handler"):
            self.viewer.register_click_handler(self._handle_click)
        self._create_visual_models()

    def notify_scene_change(self):
        self._clear_camera_linesets()
        self._ui_window = None
        self._camera_targets = {}
        self._camera_ups = {}
        self._initial_local_poses = {}
        self._initial_targets = {}
        self._initial_ups = {}
        if self._active_camera_name not in self.camera_names:
            self._active_camera_name = (
                self.camera_names[0] if self.camera_names else None
            )
        self._capture_initial_camera_state()

    def clear_scene(self):
        self._clear_camera_linesets()

    def close(self):
        self._clear_camera_linesets()
        self._line_set_model = None
        self._active_line_set_model = None
        self._ui_window = None
        self._gizmo = None
        self._target_gizmo = None
        self._camera_selector = None
        self._camera_targets = {}
        self._camera_ups = {}
        self._initial_local_poses = {}
        self._initial_targets = {}
        self._initial_ups = {}

    @property
    def editable_cameras(self) -> dict[str, Camera]:
        cameras = {}
        for uid, sensor in self.env._sensors.items():
            if isinstance(sensor, Camera):
                cameras[f"sensor:{uid}"] = sensor
        return cameras

    @property
    def camera_names(self) -> list[str]:
        return list(self.editable_cameras.keys())

    @property
    def camera_items(self) -> list[str]:
        return ["None"] + self.camera_names

    @property
    def active_camera_name(self) -> Optional[str]:
        if self._active_camera_name not in self.editable_cameras:
            self._active_camera_name = (
                self.camera_names[0] if self.camera_names else None
            )
        return self._active_camera_name

    @active_camera_name.setter
    def active_camera_name(self, value: Optional[str]):
        self._active_camera_name = value if value in self.editable_cameras else None
        if self._active_camera_name is not None:
            self._ensure_look_at_state(self._active_camera_name)
        self.viewer.notify_render_update()

    @property
    def active_camera_index(self) -> int:
        active_camera_name = self.active_camera_name
        if active_camera_name is None:
            return 0
        return self.camera_names.index(active_camera_name) + 1

    @active_camera_index.setter
    def active_camera_index(self, index: int):
        if index <= 0 or index > len(self.camera_names):
            self.active_camera_name = None
        else:
            self.active_camera_name = self.camera_names[index - 1]

    @property
    def active_camera(self) -> Optional[Camera]:
        active_camera_name = self.active_camera_name
        if active_camera_name is None:
            return None
        return self.editable_cameras[active_camera_name]

    @property
    def gizmo_matrix(self):
        if self.active_camera_name is None:
            return np.eye(4)
        self._ensure_look_at_state(self.active_camera_name)
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, 3] = self.active_eye_position
        return matrix

    @gizmo_matrix.setter
    def gizmo_matrix(self, matrix):
        if self.active_camera_name is None:
            return
        self.active_eye_position = np.asarray(matrix[:3, 3], dtype=np.float32)

    @property
    def target_gizmo_matrix(self):
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, 3] = self.active_target_position
        return matrix

    @target_gizmo_matrix.setter
    def target_gizmo_matrix(self, matrix):
        self.active_target_position = np.asarray(matrix[:3, 3], dtype=np.float32)

    @property
    def active_eye_position(self):
        active_camera = self.active_camera
        if active_camera is None:
            return np.zeros(3, dtype=np.float32)
        pose = self._get_camera_global_pose(active_camera)
        return np.asarray(pose.p, dtype=np.float32)

    @active_eye_position.setter
    def active_eye_position(self, value):
        if self.active_camera_name is None:
            return
        self._apply_look_at(self.active_camera_name, eye=np.asarray(value, dtype=np.float32))

    @property
    def active_target_position(self):
        if self.active_camera_name is None:
            return np.zeros(3, dtype=np.float32)
        self._ensure_look_at_state(self.active_camera_name)
        return self._camera_targets[self.active_camera_name].copy()

    @active_target_position.setter
    def active_target_position(self, value):
        if self.active_camera_name is None:
            return
        self._apply_look_at(
            self.active_camera_name, target=np.asarray(value, dtype=np.float32)
        )

    @property
    def active_up_vector(self):
        if self.active_camera_name is None:
            return np.array([0, 0, 1], dtype=np.float32)
        self._ensure_look_at_state(self.active_camera_name)
        return self._camera_ups[self.active_camera_name].copy()

    @active_up_vector.setter
    def active_up_vector(self, value):
        if self.active_camera_name is None:
            return
        self._apply_look_at(
            self.active_camera_name, up=np.asarray(value, dtype=np.float32)
        )

    def get_ui_windows(self):
        self._build_ui()
        if self._ui_window is None:
            return []
        return [self._ui_window]

    def after_render(self):
        if self.viewer.scene is None:
            return
        self._update_camera_linesets()

    def look_through_active_camera(self, _=None):
        active_camera = self.active_camera
        if active_camera is None:
            return
        self.viewer.set_camera_pose(self._get_camera_global_pose(active_camera))

    def set_active_camera_from_view(self, _=None):
        active_camera = self.active_camera
        if active_camera is None:
            return
        self._set_camera_global_pose(
            active_camera, self.viewer.window.get_camera_pose()
        )
        if self.active_camera_name is not None:
            self._sync_look_at_state_from_camera(
                self.active_camera_name, preserve_distance=True
            )
        self.viewer.notify_render_update()

    def copy_active_camera_config(self, _=None):
        active_camera = self.active_camera
        active_camera_name = self.active_camera_name
        if active_camera is None or active_camera_name is None:
            return

        self._ensure_look_at_state(active_camera_name)
        eye = self._format_vector(self.active_eye_position)
        target = self._format_vector(self.active_target_position)
        up = self._format_vector(self.active_up_vector)
        camera_uid = active_camera_name.split(":", 1)[1]

        config_lines = [
            f'{camera_uid}_pose = sapien_utils.look_at(eye={eye}, target={target}, up={up})',
        ]
        if active_camera.config.fov is not None:
            config_lines.append(
                'CameraConfig('
                f'"{camera_uid}", {camera_uid}_pose, {active_camera.config.width}, '
                f'{active_camera.config.height}, {active_camera.config.fov!r}, '
                f'{active_camera.config.near!r}, {active_camera.config.far!r}'
                ')'
            )
        else:
            intrinsic = np.asarray(active_camera.config.intrinsic).tolist()
            config_lines.append(
                'CameraConfig('
                f'"{camera_uid}", pose={camera_uid}_pose, width={active_camera.config.width}, '
                f'height={active_camera.config.height}, intrinsic={intrinsic}, '
                f'near={active_camera.config.near!r}, far={active_camera.config.far!r}'
                ')'
            )
        copy_to_clipboard("\n".join(config_lines))

    def reset_camera_configs(self, _=None):
        for camera_name, local_pose in self._initial_local_poses.items():
            camera = self.editable_cameras.get(camera_name)
            if camera is None:
                continue
            camera.camera.set_local_pose(local_pose)
            camera.config.pose = Pose.create(local_pose)

        self._camera_targets = {
            camera_name: value.copy()
            for camera_name, value in self._initial_targets.items()
        }
        self._camera_ups = {
            camera_name: value.copy() for camera_name, value in self._initial_ups.items()
        }
        self._active_camera_name = None
        self.viewer.select_entity(None)
        self._ui_window = None
        self._gizmo = None
        self._target_gizmo = None
        self.viewer.notify_render_update()

    def _build_ui(self):
        if self.viewer.scene is None or not self.camera_names:
            self._ui_window = None
            return

        if self._ui_window is None:
            self._gizmo = R.UIGizmo().Bind(self, "gizmo_matrix")
            self._target_gizmo = R.UIGizmo().Bind(self, "target_gizmo_matrix")
            self._camera_selector = (
                R.UIOptions()
                .Label("Camera")
                .Style("select")
                .BindItems(self, "camera_items")
                .BindIndex(self, "active_camera_index")
            )

            self._ui_window = (
                R.UIWindow()
                .Label("Camera Editor")
                .Pos(10, 420)
                .Size(420, 430)
                .append(
                    R.UICheckbox().Label("Enabled").Bind(self, "enabled"),
                    self._camera_selector,
                    R.UIDisplayText().Text(
                        "Click a camera frustum, then drag the pose gizmo to move it."
                    ),
                    R.UISameLine().append(
                        R.UIButton()
                        .Label("Look Through")
                        .Callback(self.look_through_active_camera),
                        R.UIButton()
                        .Label("Set From View")
                        .Callback(self.set_active_camera_from_view),
                        R.UIButton()
                        .Label("Reset")
                        .Callback(self.reset_camera_configs),
                        R.UIButton()
                        .Label("Copy Config")
                        .Callback(self.copy_active_camera_config),
                    ),
                    R.UIConditional()
                    .Bind(lambda: self.enabled and self.active_camera is not None)
                    .append(
                        R.UISection()
                        .Label("Position")
                        .Expanded(True)
                        .append(
                            R.UIDisplayText().Text(
                                "Move the camera position here. Orientation stays driven by Look At."
                            ),
                            self._gizmo,
                        ),
                        R.UISection()
                        .Label("Look At")
                        .Expanded(True)
                        .append(
                            R.UIDisplayText().Text(
                                "Edit eye/target/up directly, or drag the target gizmo."
                            ),
                            R.UIInputFloat3()
                            .Label("Eye")
                            .Bind(self, "active_eye_position"),
                            R.UIInputFloat3()
                            .Label("Target")
                            .Bind(self, "active_target_position"),
                            R.UIInputFloat3()
                            .Label("Up")
                            .Bind(self, "active_up_vector"),
                            self._target_gizmo,
                        ),
                    ),
                )
            )

        projection = self.viewer.window.get_camera_projection_matrix()
        view = (
            (self.viewer.window.get_camera_pose() * _VIEWER_GL_POSE)
            .inv()
            .to_transformation_matrix()
        )
        self._gizmo.CameraMatrices(view, projection)
        self._gizmo.Matrix(self.gizmo_matrix)
        self._target_gizmo.CameraMatrices(view, projection)
        self._target_gizmo.Matrix(self.target_gizmo_matrix)

    def _create_visual_models(self):
        self._line_set_model = self.viewer.renderer_context.create_line_set(
            _CAMERA_LINESET_VERTICES,
            _INACTIVE_CAMERA_LINESET_COLORS,
        )
        self._active_line_set_model = self.viewer.renderer_context.create_line_set(
            _CAMERA_LINESET_VERTICES,
            _ACTIVE_CAMERA_LINESET_COLORS,
        )

    def _clear_camera_linesets(self):
        if self.viewer.render_scene is None:
            self._line_sets = []
            self._line_set_names = []
            self._line_set_active_name = None
            return
        for node in self._line_sets:
            self.viewer.render_scene.remove_node(node)
        self._line_sets = []
        self._line_set_names = []
        self._line_set_active_name = None

    def _update_camera_linesets(self):
        if not self.enabled or self.viewer.render_scene is None:
            self._clear_camera_linesets()
            return

        camera_names = self.camera_names
        if (
            len(self._line_sets) != len(camera_names)
            or self._line_set_names != camera_names
            or self._line_set_active_name != self.active_camera_name
        ):
            self._clear_camera_linesets()
            for camera_name in camera_names:
                line_set_model = (
                    self._active_line_set_model
                    if camera_name == self.active_camera_name
                    else self._line_set_model
                )
                self._line_sets.append(
                    self.viewer.render_scene.add_line_set(line_set_model)
                )
            self._line_set_names = list(camera_names)
            self._line_set_active_name = self.active_camera_name

        for line_set, camera_name in zip(self._line_sets, camera_names):
            camera = self.editable_cameras[camera_name]
            model_matrix = self._get_camera_model_matrix(camera)
            line_set.set_position(model_matrix[:3, 3])
            line_set.set_rotation(mat2quat(model_matrix[:3, :3]))
            line_set.set_scale(
                np.array(
                    [
                        np.tan(camera.camera.fovx / 2),
                        np.tan(camera.camera.fovy / 2),
                        1.0,
                    ]
                )
                * 0.3
            )

    def _handle_click(self, _viewer, x: int, y: int) -> bool:
        if not self.enabled:
            return False

        click_target = np.array([x, y], dtype=np.float32)
        best_match = None
        for camera_name, camera in self.editable_cameras.items():
            projected = self._project_camera_origin(camera)
            if projected is None:
                continue
            distance = np.linalg.norm(projected[:2] - click_target)
            if distance > self.selection_radius_px:
                continue
            candidate = (distance, projected[2], camera_name)
            if best_match is None or candidate < best_match:
                best_match = candidate

        if best_match is None:
            return False

        self.active_camera_name = best_match[2]
        self.viewer.select_entity(None)
        return True

    def _project_camera_origin(self, camera: Camera):
        segmentation_width, segmentation_height = self.viewer.window.get_picture_size(
            "Segmentation"
        )
        pose = self._get_camera_global_pose(camera)
        point = np.array([pose.p[0], pose.p[1], pose.p[2], 1.0], dtype=np.float32)
        view = (
            (self.viewer.window.get_camera_pose() * _VIEWER_GL_POSE)
            .inv()
            .to_transformation_matrix()
        )
        clip = self.viewer.window.get_camera_projection_matrix() @ (view @ point)
        if clip[3] <= 1e-6:
            return None
        ndc = clip[:3] / clip[3]
        if ndc[2] < -1 or ndc[2] > 1:
            return None
        px = (ndc[0] * 0.5 + 0.5) * segmentation_width
        py = (1 - (ndc[1] * 0.5 + 0.5)) * segmentation_height
        return np.array([px, py, ndc[2]], dtype=np.float32)

    def _get_camera_global_pose(self, camera: Camera) -> sapien.Pose:
        pose = camera.camera.get_global_pose()
        if len(pose) > 1:
            pose = pose[0]
        return pose.sp

    def _get_camera_model_matrix(self, camera: Camera) -> np.ndarray:
        matrix = camera.camera.get_model_matrix()
        if hasattr(matrix, "detach"):
            matrix = matrix.detach().cpu().numpy()
        else:
            matrix = np.asarray(matrix)
        if matrix.ndim == 3:
            matrix = matrix[0]
        return matrix

    def _set_camera_global_pose(self, camera: Camera, pose: sapien.Pose):
        local_pose = pose
        if camera.camera.mount is not None:
            mount_pose = camera.camera.mount.pose
            if len(mount_pose) > 1:
                mount_pose = mount_pose[0]
            local_pose = mount_pose.sp.inv() * pose
        camera.camera.set_local_pose(local_pose)
        camera.config.pose = Pose.create(local_pose)

    def _capture_initial_camera_state(self):
        for camera_name, camera in self.editable_cameras.items():
            local_pose = camera.camera.get_local_pose()
            if len(local_pose) > 1:
                local_pose = local_pose[0]
            self._initial_local_poses[camera_name] = local_pose.sp
            pose = self._get_camera_global_pose(camera)
            transform = pose.to_transformation_matrix()
            eye = np.asarray(pose.p, dtype=np.float32)
            forward = np.asarray(transform[:3, 0], dtype=np.float32)
            up = self._normalize_vector(
                np.asarray(transform[:3, 2], dtype=np.float32),
                fallback=np.array([0, 0, 1], dtype=np.float32),
            )
            self._initial_targets[camera_name] = (
                eye + forward * self.default_target_distance
            )
            self._initial_ups[camera_name] = up

    def _ensure_look_at_state(self, camera_name: str):
        if camera_name in self._camera_targets and camera_name in self._camera_ups:
            return
        self._sync_look_at_state_from_camera(camera_name, preserve_distance=False)

    def _sync_look_at_state_from_camera(
        self, camera_name: str, preserve_distance: bool = True
    ):
        camera = self.editable_cameras[camera_name]
        pose = self._get_camera_global_pose(camera)
        transform = pose.to_transformation_matrix()
        eye = np.asarray(pose.p, dtype=np.float32)
        forward = np.asarray(transform[:3, 0], dtype=np.float32)
        up = np.asarray(transform[:3, 2], dtype=np.float32)

        target_distance = self.default_target_distance
        if preserve_distance and camera_name in self._camera_targets:
            target_distance = np.linalg.norm(self._camera_targets[camera_name] - eye)
            if target_distance < 1e-4:
                target_distance = self.default_target_distance

        self._camera_targets[camera_name] = eye + forward * target_distance
        self._camera_ups[camera_name] = self._normalize_vector(
            up, fallback=np.array([0, 0, 1], dtype=np.float32)
        )

    def _apply_look_at(
        self,
        camera_name: str,
        eye: Optional[np.ndarray] = None,
        target: Optional[np.ndarray] = None,
        up: Optional[np.ndarray] = None,
    ):
        self._ensure_look_at_state(camera_name)
        eye = (
            np.asarray(eye, dtype=np.float32)
            if eye is not None
            else self.active_eye_position
        )
        target = (
            np.asarray(target, dtype=np.float32)
            if target is not None
            else self._camera_targets[camera_name]
        )
        up = self._normalize_vector(
            up if up is not None else self._camera_ups[camera_name],
            fallback=self._camera_ups[camera_name],
        )
        if np.linalg.norm(target - eye) < 1e-4:
            return

        self._camera_targets[camera_name] = target
        self._camera_ups[camera_name] = up
        pose = sapien_utils.look_at(eye=eye, target=target, up=up).sp
        self._set_camera_global_pose(self.editable_cameras[camera_name], pose)
        self.viewer.notify_render_update()

    def _normalize_vector(self, vector, fallback: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm < 1e-6:
            return np.asarray(fallback, dtype=np.float32)
        return vector / norm

    def _format_vector(self, vector: np.ndarray) -> str:
        values = np.asarray(vector, dtype=np.float32).tolist()
        return "[" + ", ".join(f"{value:.6g}" for value in values) + "]"
