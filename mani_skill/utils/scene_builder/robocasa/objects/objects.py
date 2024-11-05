import numpy as np
import sapien
from transforms3d.euler import quat2mat

from mani_skill.utils.scene_builder.robocasa.fixtures.mujoco_object import MujocoObject
from mani_skill.utils.scene_builder.robocasa.utils.mjcf_utils import string_to_array


class MJCFObject(MujocoObject):
    """
    Blender object with support for changing the scaling
    """

    def __init__(
        self,
        scene,
        name,
        mjcf_path,
        scale=1.0,
        solimp=(0.998, 0.998, 0.001),
        solref=(0.001, 1),
        density=100,
        friction=(0.95, 0.3, 0.1),
        margin=None,
        rgba=None,
        priority=None,
    ):
        # get scale in x, y, z
        if isinstance(scale, float):
            scale = [scale, scale, scale]
        elif isinstance(scale, tuple) or isinstance(scale, list):
            assert len(scale) == 3
            scale = tuple(scale)
        else:
            raise Exception("got invalid scale: {}".format(scale))
        scale = np.array(scale)

        # note (stao): the values below are unused atm
        self.solimp = solimp
        self.solref = solref
        self.density = density
        self.friction = friction
        self.margin = margin

        self.priority = priority

        self.rgba = rgba

        super().__init__(
            scene=scene,
            xml=mjcf_path,
            name=name,
            # joints=[dict(type="free", damping="0.0005")],
            # obj_type="all",
            # duplicate_collision_geoms=False,
            scale=scale,
        )

    def build(self, scene_idxs: list[int]):
        self.actor_builder.set_scene_idxs(scene_idxs)
        self.actor_builder.initial_pose = sapien.Pose(p=self.pos, q=self.quat)
        self.actor = self.actor_builder.build_dynamic(
            name=self.name + f"_{scene_idxs[0]}"
        )
        return self

    @property
    def horizontal_radius(self):
        horizontal_radius_site = self.loader.xml.find(
            ".//site[@name='{}horizontal_radius_site']".format(self.naming_prefix)
        )
        site_values = string_to_array(horizontal_radius_site.get("pos"))
        return np.linalg.norm(site_values[0:2])

    def get_bbox_points(self, trans=None, rot=None):
        """
        Get the full 8 bounding box points of the object
        rot: a rotation matrix
        """
        bbox_offsets = []

        bottom_offset = self.bottom_offset
        top_offset = self.top_offset
        horizontal_radius_site = self.loader.xml.find(
            ".//site[@name='{}horizontal_radius_site']".format(self.naming_prefix)
        )
        horiz_radius = string_to_array(horizontal_radius_site.get("pos"))[:2]

        center = np.mean([bottom_offset, top_offset], axis=0)
        half_size = [horiz_radius[0], horiz_radius[1], top_offset[2] - center[2]]

        bbox_offsets = [
            center + half_size * np.array([-1, -1, -1]),  # p0
            center + half_size * np.array([1, -1, -1]),  # px
            center + half_size * np.array([-1, 1, -1]),  # py
            center + half_size * np.array([-1, -1, 1]),  # pz
            center + half_size * np.array([1, 1, 1]),
            center + half_size * np.array([-1, 1, 1]),
            center + half_size * np.array([1, -1, 1]),
            center + half_size * np.array([1, 1, -1]),
        ]

        if trans is None:
            trans = np.array([0, 0, 0])
        if rot is not None:
            rot = quat2mat(rot)
        else:
            rot = np.eye(3)

        points = [(np.matmul(rot, p) + trans) for p in bbox_offsets]
        return points
