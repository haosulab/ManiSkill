from typing import Dict, List, Union, Optional

import numpy as np
import cv2
import open3d as o3d

from ..logger import get_logger
from ..lib3d import np2pcd
from .cv2_visualizer import CV2Visualizer
from .o3d_gui_visualizer import O3DGUIVisualizer
from .utils import draw_mask, colorize_mask
try:
    from pynput.keyboard import Key, KeyCode, Listener
except ImportError as e:
    get_logger("visualizer.py").warning(f"ImportError: {e}")


pause_render = False


def _on_key_press(key):
    if key == KeyCode.from_char('p'):
        global pause_render
        pause_render = True


class Visualizer:
    def __init__(self):
        self.cv2_vis = CV2Visualizer()
        self.o3d_vis = O3DGUIVisualizer()

        self.key_listener = Listener(on_press=_on_key_press)
        self.key_listener.start()

    def reset(self, **observations):
        self.cv2_vis.clear_image()
        self.o3d_vis.clear_geometries()

        if len(observations) > 0:
            self.show_observation(**observations)
            self.render()

    def show_observation(self, camera_names: Optional[List[str]] = None,
                         **obs_dict: Dict[str, Union[np.ndarray,
                                                     List[np.ndarray],
                                                     o3d.geometry.Geometry]]):
        """Render observations
        :param camera_names: camera names if obs_data are from multiple cameras
                             The order should match with order of obs_data
        :param obs_dict: dict, {obs_name: obs_data}
                         obs_name must contain one of
                         ['color_image', 'depth_image', 'mask',
                          'xyz_image', 'points', 'pts', 'bbox', 'mesh']
                         obs_data can be np.ndarray from one camera or
                         a list of np.ndarray from multiple cameras
        """
        images = {}  # {name_with_group: image}
        o3d_geometries = {}  # {name_with_group: geometry}

        color_images = []
        for obs_name, obs_data in obs_dict.items():
            if not isinstance(obs_data, list):
                obs_name, obs_data = [obs_name], [obs_data]
            else:
                # Prepend camera_name to obs_name
                obs_name = [f"{cam_name}/{obs_name}"
                            for cam_name in camera_names]

            for i, (name, obs) in enumerate(zip(obs_name, obs_data)):
                if "color_image" in name:  # color image
                    images[name] = obs
                    color_images.append(obs)
                elif "depth_image" in name:  # depth image
                    images[name] = obs
                elif "mask" in name:  # mask images
                    if len(color_images) > 0:
                        images[name+"_overlay"] = draw_mask(
                            color_images[i].copy(), obs
                        )
                    images[name] = colorize_mask(obs)
                elif "xyz_image" in name:  # xyz_image
                    colors = None
                    if len(color_images) > 0:
                        colors = color_images[i].reshape(-1, 3) / 255.0
                    o3d_geometries[name] = np2pcd(obs.reshape(-1, 3), colors)
                elif "points" in name or "pts" in name:  # point clouds
                    o3d_geometries[name] = np2pcd(obs.reshape(-1, 3))
                elif "bbox" in name:  # bounding boxes
                    assert isinstance(obs,
                                      (o3d.geometry.AxisAlignedBoundingBox,
                                       o3d.geometry.OrientedBoundingBox)), \
                        f"Not a bbox: {type(obs) = }"
                    o3d_geometries[name] = obs
                elif "mesh" in name:  # TriangleMesh
                    assert isinstance(obs,
                                      o3d.geometry.TriangleMesh), \
                        f"Not a mesh: {type(obs) = }"
                    o3d_geometries[name] = obs
                else:
                    raise NotImplementedError(f"Unknown object {name = }")

        # Sort images based on key
        self.cv2_vis.show_images([img for _, img in sorted(images.items())])
        for name, geometry in o3d_geometries.items():
            self.o3d_vis.add_geometry(name, geometry, show=True)# show="xyz_image" in name)

    #@staticmethod
    #def update_aabb_geometry(aabb_dst, aabb_src):
    #    aabb_dst.min_bound = aabb_src.min_bound
    #    aabb_dst.max_bound = aabb_src.max_bound
    #    return aabb_dst

    #def update_object(self, color_image, depth_image, pred_masks=None,
    #                  xyz_image=None, cube_pts=None, bowl_pts=None):
    #    if pred_masks is not None:
    #        # OpenCV
    #        depth_colormap = cv2.applyColorMap(
    #            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
    #        )
    #        vis_image = np.hstack([cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR),
    #                            depth_colormap])
    #        vis_image = np.vstack([
    #            vis_image,
    #            cv2.cvtColor(
    #                np.hstack([draw_mask(color_image, pred_masks),
    #                        colorize_mask(pred_masks)]),
    #            cv2.COLOR_RGB2BGR)
    #        ])
    #        cv2.imshow("Color / Depth", vis_image)

    #        # Open3D
    #        scene_pcd = self.o3d_geometries["scene_pcd"]
    #        scene_pcd.points = o3d.utility.Vector3dVector(xyz_image.reshape(-1, 3))
    #        scene_pcd.colors = o3d.utility.Vector3dVector(color_image.reshape(-1, 3) / 255.0)
    #        cube_aabb = self.o3d_geometries["cube_aabb"]
    #        cube_aabb = self.update_aabb_geometry(
    #            cube_aabb, np2pcd(cube_pts).get_axis_aligned_bounding_box()
    #        )
    #        bowl_aabb = self.o3d_geometries["bowl_aabb"]
    #        bowl_aabb = self.update_aabb_geometry(
    #            bowl_aabb, np2pcd(bowl_pts).get_axis_aligned_bounding_box()
    #        )

    #        for geometry in self.o3d_geometries.values():
    #            self.o3d_vis.update_geometry(geometry)

    #        self.render()
    #    else:
    #        vis_image = np.hstack([
    #            cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR),
    #            cv2.cvtColor(depth_image, cv2.COLOR_RGB2BGR)
    #        ])
    #        cv2.imshow("Color / Depth", vis_image)
    #        cv2.waitKey(1)

    def render(self):
        global pause_render
        if pause_render:
            self.o3d_vis.toggle_pause(True)
            pause_render = False

        # Render visualizer
        # self.o3d_vis.render() returns only when not paused or single_step
        self.cv2_vis.render()
        self.o3d_vis.render(render_step_fn=self.cv2_vis.render)

    def close(self):
        self.cv2_vis.close()
        self.o3d_vis.close()

    def __del__(self):
        self.key_listener.stop()
        self.close()
