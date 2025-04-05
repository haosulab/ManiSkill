import sapien
from sapien.utils import Viewer
import sys

from mani_skill.render import SAPIEN_RENDER_SYSTEM
from mani_skill.sensors.camera import CameraConfig


def create_viewer(viewer_camera_config: CameraConfig):
    """Creates a viewer with the given camera config"""
    if SAPIEN_RENDER_SYSTEM == "3.0":
        sapien.render.set_viewer_shader_dir(
            viewer_camera_config.shader_config.shader_pack
        )
        if viewer_camera_config.shader_config.shader_pack[:2] == "rt":
            sapien.render.set_ray_tracing_denoiser(
                viewer_camera_config.shader_config.shader_pack_config[
                    "ray_tracing_denoiser"
                ]
            )
            sapien.render.set_ray_tracing_path_depth(
                viewer_camera_config.shader_config.shader_pack_config[
                    "ray_tracing_path_depth"
                ]
            )
            sapien.render.set_ray_tracing_samples_per_pixel(
                viewer_camera_config.shader_config.shader_pack_config[
                    "ray_tracing_samples_per_pixel"
                ]
            )
        viewer = Viewer(
            resolutions=(viewer_camera_config.width, viewer_camera_config.height)
        )
        if sys.platform == 'darwin':  # macOS
            viewer.window.set_content_scale(1)
    elif SAPIEN_RENDER_SYSTEM == "3.1":
        # TODO (stao): figure out how shader pack configs can be set at run time
        viewer = Viewer(
            resolutions=(viewer_camera_config.width, viewer_camera_config.height),
            shader_pack=sapien.render.get_shader_pack(
                viewer_camera_config.shader_config.shader_pack
            ),
        )
        

    return viewer
