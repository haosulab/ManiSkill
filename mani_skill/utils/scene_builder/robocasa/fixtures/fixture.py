import abc
from enum import IntEnum

import numpy as np
import sapien
from transforms3d.euler import euler2mat, quat2euler, quat2mat

from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.scene_builder.robocasa.fixtures.mujoco_object import MujocoObject
from mani_skill.utils.scene_builder.robocasa.utils.object_utils import (
    get_pos_after_rel_offset,
)


class FixtureType(IntEnum):
    """
    Enum for fixture types in robocasa kitchen environments.
    """

    COUNTER = 1
    MICROWAVE = 2
    STOVE = 3
    SINK = 4
    CABINET = 5
    DRAWER = 6
    SHELF = 7
    COFFEE_MACHINE = 8
    DOOR = 9
    DOOR_HINGE = 10
    DOOR_HINGE_SINGLE = 11
    DOOR_HINGE_DOUBLE = 12
    DOOR_TOP_HINGE = 13
    DOOR_TOP_HINGE_SINGLE = 14
    DOOR_TOP_HINGE_DOUBLE = 15
    CABINET_TOP = 16
    TOASTER = 17
    DINING_COUNTER = 18
    TOP_DRAWER = 19
    STOOL = 20
    ISLAND = 21
    COUNTER_NON_CORNER = 22


class Fixture(MujocoObject):
    def __init__(
        self,
        scene: ManiSkillScene,
        xml,
        name,
        duplicate_collision_geoms=True,
        pos=None,
        scale=1,
        size=None,
        placement=None,
        rng=None,
    ):
        super().__init__(scene, xml, name, pos=pos, scale=scale)
        if pos is not None:
            self.set_pos(pos)

        # scale based on specified max dimension
        self.size = np.array([self.width, self.depth, self.height])
        if size is not None:
            self.set_scale_from_size(size)
        # based on exterior points, overwritten by subclasses (e.g. Counter) that do not have such sites

        # set offset between center of object and center of exterior bounding boxes
        if self.width is not None:
            try:
                # calculate based on bounding points
                p0 = self._bounds_sites["ext_p0"]
                px = self._bounds_sites["ext_px"]
                py = self._bounds_sites["ext_py"]
                pz = self._bounds_sites["ext_pz"]
                self.origin_offset = np.array(
                    [
                        np.mean((p0[0], px[0])),
                        np.mean((p0[1], py[1])),
                        np.mean((p0[2], pz[2])),
                    ]
                )
            except KeyError:
                self.origin_offset = [0, 0, 0]
        else:
            self.origin_offset = [0, 0, 0]
        self.origin_offset = np.array(self.origin_offset)

        # placement config, for determining where to place fixture (most fixture will not use this)
        self._placement = placement

        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()

    @property
    def is_articulation(self):
        return hasattr(self, "articulation_builder")

    def build(self, scene_idxs: list[int]):
        if self.is_articulation:
            self.articulation_builder.initial_pose = sapien.Pose(
                p=self.pos, q=self.quat
            )
            self.articulation_builder.set_scene_idxs(scene_idxs)
            self.articulation = self.articulation_builder.build(
                name=self.name + f"_{scene_idxs[0]}", fix_root_link=True
            )
        else:
            self.actor_builder.set_scene_idxs(scene_idxs)
            self.actor_builder.initial_pose = sapien.Pose(p=self.pos, q=self.quat)
            self.actor = self.actor_builder.build_static(
                name=self.name + f"_{scene_idxs[0]}"
            )
        return self

    """Functions from RoboCasa Fixture class"""

    def set_origin(self, origin):
        """
        Set the origin of the fixture to a specified position

        Args:
            origin (3-tuple): new (x, y, z) position of the fixture
        """
        # compute new position
        fixture_rot = np.array([0, 0, self.rot])
        from transforms3d.euler import euler2mat

        fixture_mat = euler2mat(*fixture_rot)
        # fixture_mat = T.euler2mat(fixture_rot)
        pos = origin + np.dot(fixture_mat, -self.origin_offset)
        self.set_pos(pos)

    def set_scale_from_size(self, size):
        """
        Set the scale of the fixture based on the desired size. If any of the dimensions are None,
        the scaling factor will be the same as one of the other two dimensions

        Args:
            size (3-tuple): (width, depth, height) of the fixture
        """
        # check that the argument is valid
        assert len(size) == 3

        # calculate and set scale according to specification
        scale = [None, None, None]
        cur_size = [self.width, self.depth, self.height]
        for (i, t) in enumerate(size):
            if t is not None:
                scale[i] = t / cur_size[i]

        scale[0] = scale[0] or scale[2] or scale[1]
        scale[1] = scale[1] or scale[0] or scale[2]
        scale[2] = scale[2] or scale[0] or scale[1]
        self.set_scale(scale)

    def get_reset_regions(self, *args, **kwargs):
        """
        returns dictionary of reset regions, each region defined as position, x_bounds, y_bounds
        """
        p0, px, py, pz = self.get_int_sites()
        return {
            "bottom": {
                "offset": (np.mean((p0[0], px[0])), np.mean((p0[1], py[1])), p0[2]),
                "size": (px[0] - p0[0], py[1] - p0[1]),
            }
        }

    def sample_reset_region(self, *args, **kwargs):
        regions = self.get_reset_regions(*args, **kwargs)
        return self.rng.choice(list(regions.values()))

    def get_site_info(self, sim):
        """
        returns user defined sites (eg. the table top, door handle sites, handle sites, shelf sites, etc)
        requires sim as position of sites can change during simulation.
        """
        info = {}
        for s in self._sites:
            name = "{}{}".format(self.naming_prefix, s)
            info[name] = sim.data.get_site_xpos(name)
        return info

    @abc.abstractmethod
    def get_state(self):
        """
        get the current state of the fixture
        """
        return

    @abc.abstractmethod
    def update_state(self, env):
        """
        update internal state of fixture
        """
        return

    # @property
    # def pos(self):
    #     return _parse_vec(self._obj.get("pos"))

    # @property
    # def quat(self):
    #     quat = self._obj.get("quat")
    #     if quat is None:
    #         # no rotation applied
    #         quat = "0 0 0 0"
    #     return _parse_vec(quat)

    @property
    def euler(self):
        euler = quat2euler(self.quat)
        if euler is None:
            # no rotation applied
            euler = [0, 0, 0]
        return np.array(euler)

    @property
    def horizontal_radius(self):
        """
        override the default behavior of only looking at first dimension for radius
        """
        horizontal_radius_site = self.worldbody.find(
            "./body/site[@name='{}horizontal_radius_site']".format(self.naming_prefix)
        )
        site_values = _parse_vec(horizontal_radius_site.get("pos"))
        return np.linalg.norm(site_values[0:2])

    @property
    def bottom_offset(self):
        return self._bounds_sites["ext_p0"]

    @property
    def width(self):
        """
        for getting the width of an object as defined by its exterior sites.
        takes scaling into account
        """
        if "ext_px" in self._bounds_sites:
            ext_p0 = self._bounds_sites["ext_p0"]
            ext_px = self._bounds_sites["ext_px"]
            w = ext_px[0] - ext_p0[0]
            return w
        else:
            return None

    @property
    def depth(self):
        """
        for getting the depth of an object as defined by its exterior sites.
        takes scaling into account
        """
        if "ext_py" in self._bounds_sites:
            ext_p0 = self._bounds_sites["ext_p0"]
            ext_py = self._bounds_sites["ext_py"]
            d = ext_py[1] - ext_p0[1]
            return d
        else:
            return None

    @property
    def height(self):
        """
        for getting the height of an object as defined by its exterior sites.
        takes scaling into account
        """
        if "ext_pz" in self._bounds_sites:
            ext_p0 = self._bounds_sites["ext_p0"]
            ext_pz = self._bounds_sites["ext_pz"]
            h = ext_pz[2] - ext_p0[2]
            return h
        else:
            return None

    def set_bounds_sites(self, pos_dict):
        """
        Set the positions of the exterior and interior bounding box sites of the object

        Args:
            pos_dict (dict): Dictionary of sites and their new positions
        """
        for (name, pos) in pos_dict.items():
            self._bounds_sites[name] = pos

    def get_ext_sites(self, all_points=False, relative=True):
        """
        Get the exterior bounding box points of the object

        Args:
            all_points (bool): If True, will return all 8 points of the bounding box

            relative (bool): If True, will return the points relative to the object's position

        Returns:
            list: 4 or 8 points
        """
        sites = [
            (self._bounds_sites["ext_p0"]),
            (self._bounds_sites["ext_px"]),
            (self._bounds_sites["ext_py"]),
            (self._bounds_sites["ext_pz"]),
        ]

        if all_points:
            p0, px, py, pz = sites
            sites += [
                np.array([p0[0], py[1], pz[2]]),
                np.array([px[0], py[1], pz[2]]),
                np.array([px[0], py[1], p0[2]]),
                np.array([px[0], p0[1], pz[2]]),
            ]

        if relative is False:
            sites = [get_pos_after_rel_offset(self, offset) for offset in sites]

        return sites

    def get_int_sites(self, all_points=False, relative=True):
        """
        Get the interior bounding box points of the object

        Args:
            all_points (bool): If True, will return all 8 points of the bounding box

            relative (bool): If True, will return the points relative to the object's position

        Returns:
            list: 4 or 8 points
        """
        sites = [
            (self._bounds_sites["int_p0"]),
            (self._bounds_sites["int_px"]),
            (self._bounds_sites["int_py"]),
            (self._bounds_sites["int_pz"]),
        ]

        if all_points:
            p0, px, py, pz = sites
            sites += [
                np.array([p0[0], py[1], pz[2]]),
                np.array([px[0], py[1], pz[2]]),
                np.array([px[0], py[1], p0[2]]),
                np.array([px[0], p0[1], pz[2]]),
            ]

        if relative is False:
            sites = [get_pos_after_rel_offset(self, offset) for offset in sites]

        return sites

    def get_bbox_points(self, trans=None, rot=None):
        """
        Get the full set of bounding box points of the object
        rot: a rotation matrix
        """
        bbox_offsets = self.get_ext_sites(all_points=True, relative=True)

        if trans is None:
            trans = self.pos
        if rot is not None:
            rot = quat2mat(rot)
        else:
            rot = euler2mat(0, 0, self.rot)

        points = [(np.matmul(rot, p) + trans) for p in bbox_offsets]
        return points
