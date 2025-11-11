from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional
import os

import numpy as np
import torch
import sapien
import numpy as np
from sapien.render import RenderBodyComponent
from transforms3d.euler import euler2quat

from mani_skill.sensors.camera import CameraConfig
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.building import actors
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.actor import Actor


def _get_random_color(color_range: tuple):
    assert (len(color_range) == 2) and (len(color_range[0]) == 3) and (len(color_range[1]) == 3), "color_range must be a tuple of two tuples of three floats"
    return np.random.uniform(*color_range).tolist() + [1]

def _get_random_texture(texture_dir: str):
    texture_files = [f for f in os.listdir(texture_dir) if f.endswith('.png')]
    texture_file = np.random.choice(texture_files)
    return sapien.render.RenderTexture2D(filename=os.path.join(texture_dir, texture_file))


@dataclass
class DistractionSet:
    """
    Factor of Variation | Description
    ---------------------------------
    MO color            | Modifies the color of the MO
    RO color            | Modifies the color of the RO (if applicable)
    MO texture          | Modifies the texture applied to the MO
    RO texture          | Modifies the texture applied to the RO (if applicable)
    MO size             | Scales the MO by a given factor
    RO size             | Scales the RO (if applicable) by a given factor
    Table color         | Modifies the color of the tabletop of the robot setup
    Light color         | Modifies the color of the lights setup in the scene.
    Table texture       | Modifies the texture applied to the tabletop of the robot setup.
    Distractor object   | Spawns a random object in the workspace of the robot.
    Background texture  | Modifies the textures applied to the walls of the scene.
    Camera pose         | Randomly perturbs the pose of a camera.

    from https://robot-colosseum.readthedocs.io/en/latest/overview.html
    """
    MO_color_cfg: dict = field(default_factory=dict)
    RO_color_cfg: dict = field(default_factory=dict)
    MO_texture_cfg: dict = field(default_factory=dict)
    RO_texture_cfg: dict = field(default_factory=dict)
    MO_size_cfg: dict = field(default_factory=dict)
    RO_size_cfg: dict = field(default_factory=dict)
    table_color_cfg: dict = field(default_factory=dict)
    light_color_cfg: dict = field(default_factory=dict)
    table_texture_cfg: dict = field(default_factory=dict)
    distractor_object_cfg: dict = field(default_factory=dict)
    background_texture_cfg: dict = field(default_factory=dict)
    camera_pose_cfg: dict = field(default_factory=dict)

    unimplemented = {
        "MO_size",
        "RO_size",
        "light_color",
        "background_texture"
    }

    def get_partial_copy(self, keys: list[str]) -> "DistractionSet":
        return DistractionSet(**{k: v for k, v in self.__dict__.items() if k in keys})

    def MO_color_enabled(self) -> bool:
        return len(self.MO_color_cfg) > 0

    def RO_color_enabled(self) -> bool:
        return len(self.RO_color_cfg) > 0

    def MO_texture_enabled(self) -> bool:
        return len(self.MO_texture_cfg) > 0

    def RO_texture_enabled(self) -> bool:
        return len(self.RO_texture_cfg) > 0

    def MO_size_enabled(self) -> bool:
        return len(self.MO_size_cfg) > 0

    def RO_size_enabled(self) -> bool:
        return len(self.RO_size_cfg) > 0

    def table_color_enabled(self) -> bool:
        return len(self.table_color_cfg) > 0

    def light_color_enabled(self) -> bool:
        return len(self.light_color_cfg) > 0

    def table_texture_enabled(self) -> bool:
        return len(self.table_texture_cfg) > 0

    def distractor_object_enabled(self) -> bool:
        return len(self.distractor_object_cfg) > 0

    def background_texture_enabled(self) -> bool:
        return len(self.background_texture_cfg) > 0

    def camera_pose_enabled(self) -> bool:
        return len(self.camera_pose_cfg) > 0

    def which_enabled_str(self) -> tuple[list[str], list[str]]:
        enabled_strs = []
        disabled_strs = []
        for k in [attr for attr in dir(self) if not attr.startswith('_')]:
            if k.endswith('_enabled') and hasattr(self, k):
                enabled_fn = getattr(self, k)
                if enabled_fn():
                    enabled_strs.append(k[:-8]) # Remove '_enabled' suffix and append
                else:
                    disabled_strs.append(k[:-8])
        return enabled_strs, disabled_strs

    def __post_init__(self):

        self._internal = {}
        for key in [
            "MO_color_cfg",
            "RO_color_cfg",
            "MO_texture_cfg",
            "RO_texture_cfg",
            "MO_size_cfg",
            "RO_size_cfg",
            "table_color_cfg",
            "light_color_cfg",
            "table_texture_cfg",
            "distractor_object_cfg",
            "background_texture_cfg",
            "camera_pose_cfg",
        ]:
            self._internal[key] = {}

        for k in self.which_enabled_str()[0]:
            assert k not in self.unimplemented, f"Distractor {k} is enabled but in the unimplemented set"

        def assert_range_correct(range: tuple):
            assert len(range) == 2, "range must be a tuple of two values"
            assert len(range[0]) == 3, "range[0] must be a tuple of three values"
            assert len(range[1]) == 3, "range[1] must be a tuple of three values"
            assert range[0][0] <= range[1][0], "range[0][0] must be less than range[1][0]"
            assert range[0][1] <= range[1][1], "range[0][1] must be less than range[1][1]"
            assert range[0][2] <= range[1][2], "range[0][2] must be less than range[1][2]"

        if self.camera_pose_enabled():
            assert_range_correct(self.camera_pose_cfg["rpy_range"])
            assert_range_correct(self.camera_pose_cfg["xyz_range"])
        if self.distractor_object_enabled():
            assert_range_correct(self.distractor_object_cfg["color_range"])
        if self.table_color_enabled():
            assert_range_correct(self.table_color_cfg["color_range"])

    def to_dict(self):
        return dict(
            MO_color_cfg=self.MO_color_cfg,
            RO_color_cfg=self.RO_color_cfg,
            MO_texture_cfg=self.MO_texture_cfg,
            RO_texture_cfg=self.RO_texture_cfg,
            MO_size_cfg=self.MO_size_cfg,
            RO_size_cfg=self.RO_size_cfg,
            table_color_cfg=self.table_color_cfg,
            light_color_cfg=self.light_color_cfg,
            table_texture_cfg=self.table_texture_cfg,
            distractor_object_cfg=self.distractor_object_cfg,
            background_texture_cfg=self.background_texture_cfg,
            camera_pose_cfg=self.camera_pose_cfg,
        )

    def update_camera_configs(self, cfgs: list[CameraConfig]) -> list[CameraConfig]:
        if not self.camera_pose_enabled():
            return cfgs

        rpy_range = self.camera_pose_cfg["rpy_range"]
        xyz_range = self.camera_pose_cfg["xyz_range"]

        for cfg in cfgs:
            rpy = np.random.uniform(*rpy_range)
            xyz = np.random.uniform(*xyz_range)
            delta_pose = sapien.Pose(p=xyz, q=euler2quat(rpy[0], rpy[1], rpy[2]))
            cfg.pose *= delta_pose

        return cfgs

    def set_color_or_texture(self, actor: Actor, color_cfg: dict, texture_cfg: dict, set_color: bool, set_texture: bool):

        if not set_color and not set_texture:
            return

        # The following code is borrowed from here: https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/domain_randomization.html
        for obj in actor._objs:
            # modify the i-th object which is in parallel environment i
            render_body_component: RenderBodyComponent = obj.find_component_by_type(RenderBodyComponent)
            for render_shape in render_body_component.render_shapes:
                for part in render_shape.parts:
                    # part.material: sapien.core.pysapien.RenderMaterial
                    if set_texture:
                        texture = _get_random_texture(texture_cfg["textures_directory"])
                    if set_color:
                        color = _get_random_color(color_cfg["color_range"])

                    if set_color and not set_texture:
                        part.material.set_base_color(color)
                    elif not set_color and set_texture:
                        part.material.set_base_color_texture(texture)
                    elif set_color and set_texture:
                        if np.random.random() < 0.5:
                            part.material.set_base_color(color)
                        else:
                            part.material.set_base_color_texture(texture)


    def load_scene_hook(self, scene: ManiSkillScene, manipulation_object: Optional[Actor], table: Optional[Actor], receiving_object: Actor | None = None):
        """
        This function is called when the scene is loaded.
        Args:
            scene (ManiSkillScene): The scene to modify.
            manipulation_object (Optional[Actor]): The manipulation object to modify. Note that this is a wrapper around
                                                    a sapien.Entity.
        """

        # New distractor spheres
        if self.distractor_object_enabled():
            n_spheres = self.distractor_object_cfg["n_spheres"]
            radius_range = self.distractor_object_cfg["radius_range"]
            color_range = self.distractor_object_cfg["color_range"]
            radii = np.random.uniform(*radius_range, size=n_spheres)

            self._internal["distractor_object_cfg"]["internal__radii"] = radii
            self._internal["distractor_object_cfg"]["internal__spheres"] = [
                actors.build_sphere(
                    scene,
                    initial_pose=sapien.Pose(),
                    name=f"distractor_sphere_{i}",
                    radius=radii[i],
                    color=np.random.uniform(*color_range).tolist() + [1.0], # alpha=1.0
                )
                for i in range(n_spheres)
            ]

        # Set table color and texture
        if table is not None:
            self.set_color_or_texture(table, self.table_color_cfg, self.table_texture_cfg, self.table_color_enabled(), self.table_texture_enabled())

        # Manipulation object
        if manipulation_object is not None:
            self.set_color_or_texture(manipulation_object, self.MO_color_cfg, self.MO_texture_cfg, self.MO_color_enabled(), self.MO_texture_enabled())

        # Receiving object
        if receiving_object is not None:
            self.set_color_or_texture(receiving_object, self.RO_color_cfg, self.RO_texture_cfg, self.RO_color_enabled(), self.RO_texture_enabled())


    def initialize_episode_hook(self, n_envs: int, mo_pose: torch.Tensor, ro_pose: torch.Tensor | None = None):
        assert mo_pose.shape[0] == n_envs
        assert mo_pose.shape[1] >= 2, f"mo_pose must have at least 2 dimensions, got {mo_pose.shape[1]}"
        if ro_pose is not None:
            assert ro_pose.shape[0] == n_envs
            assert ro_pose.shape[1] >= 2, f"ro_pose must have at least 2 dimensions, got {ro_pose.shape[1]}"

        # TODO: Make sure that the sampled poses are beyond some epsilon of RO/ro objects

        if self.distractor_object_enabled():

            x_lims = self.distractor_object_cfg["x_lims"]
            y_lims = self.distractor_object_cfg["y_lims"]
            radii = self._internal["distractor_object_cfg"]["internal__radii"]
            x_range = x_lims[1] - x_lims[0]
            y_range = y_lims[1] - y_lims[0]

            # What happens if you set the poses such that the spheres collide with one another?
            for i, sphere in enumerate(self._internal["distractor_object_cfg"]["internal__spheres"]):
                xyz = torch.rand((n_envs, 3), dtype=torch.float32)
                xyz[:, 0] = x_range * xyz[:, 0] + x_lims[0]
                xyz[:, 1] = y_range * xyz[:, 1] + y_lims[0]
                xyz[:, 2] = radii[i]
                sphere.set_pose(Pose.create_from_pq(p=xyz))


_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "../assets")
all_distractor_set = DistractionSet(
    distractor_object_cfg={
        "n_spheres": 1,
        "radius_range": (0.01, 0.02),"color_range": ((0, 0, 0), (1, 1, 1)),
        "x_lims": (-0.1, 0.1),
        "y_lims": (-0.1, 0.1),
    },
    MO_color_cfg ={
        "color_range": ((0, 0, 0), (1, 1, 1)),
    },
    MO_texture_cfg = {
        "textures_directory": os.path.join(_ASSETS_DIR, "textures"),
    },
    table_color_cfg = {
        "color_range": ((0, 0, 0), (1, 1, 1)),
    },
    table_texture_cfg = {
        "textures_directory": os.path.join(_ASSETS_DIR, "textures"),
    },
    camera_pose_cfg = {
        "rpy_range": ((-0.035, -0.035, -0.035), (0.035, 0.035, 0.035)), # aproximately 2 degrees
        "xyz_range": ((-0.025, -0.025, 0.025), (0.025, 0.025, 0.025)),        # 2.5 cm
    }
)

DISTRACTION_SETS = {
    "none".upper(): DistractionSet(),
    "all".upper(): DistractionSet(
        distractor_object_cfg={
            "n_spheres": 1,
            "radius_range": (0.01, 0.02),
            "color_range": ((0, 0, 0), (1, 1, 1)),
            "x_lims": (-0.1, 0.1),
            "y_lims": (-0.1, 0.1),
        },
        MO_color_cfg ={
            "color_range": ((0, 0, 0), (1, 1, 1)),
        },
        MO_texture_cfg = {
            "textures_directory": os.path.join(_ASSETS_DIR, "textures"),
        },
        RO_color_cfg ={
            "color_range": ((0, 0, 0), (1, 1, 1)),
        },
        RO_texture_cfg = {
            "textures_directory": os.path.join(_ASSETS_DIR, "textures"),
        },
        table_color_cfg = {
            "color_range": ((0, 0, 0), (1, 1, 1)),
        },
        table_texture_cfg = {
            "textures_directory": os.path.join(_ASSETS_DIR, "textures"),
        },
        camera_pose_cfg = {
            "rpy_range": ((-0.035, -0.035, -0.035), (0.035, 0.035, 0.035)), # aproximately 2 degrees
            "xyz_range": ((-0.025, -0.025, 0.025), (0.025, 0.025, 0.025)),        # 2.5 cm
        }
    ),
    "distractor_object_cfg".upper(): all_distractor_set.get_partial_copy(["distractor_object_cfg"]),
    "MO_color_cfg".upper(): all_distractor_set.get_partial_copy(["MO_color_cfg"]),
    "MO_texture_cfg".upper(): all_distractor_set.get_partial_copy(["MO_texture_cfg"]),
    "RO_color_cfg".upper(): all_distractor_set.get_partial_copy(["RO_color_cfg"]),
    "RO_texture_cfg".upper(): all_distractor_set.get_partial_copy(["RO_texture_cfg"]),
    "table_color_cfg".upper(): all_distractor_set.get_partial_copy(["table_color_cfg"]),
    "table_texture_cfg".upper(): all_distractor_set.get_partial_copy(["table_texture_cfg"]),
    "camera_pose_cfg".upper(): all_distractor_set.get_partial_copy(["camera_pose_cfg"]),
}