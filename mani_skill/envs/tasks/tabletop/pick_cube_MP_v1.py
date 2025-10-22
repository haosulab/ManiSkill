import numpy as np
from typing import List, Optional, Union
import torch
import sapien

import mani_skill.envs.utils.randomization as randomization
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building import actors
from mani_skill.envs.distraction_set import DistractionSet
from mani_skill.envs.tasks.tabletop.get_camera_config import get_camera_configs, get_human_render_camera_config
from mani_skill.utils.building.actors.ycb import get_ycb_builder


@register_env("PickCubeMP-v1", max_episode_steps=100)
class PickCubeMPEnv(PickCubeEnv):
    """
    """

    # The following are copied from place_sphere.py:
    inner_side_half_len = 0.02  # side length of the bin's inner square
    short_side_half_size = 0.0025  # length of the shortest edge of the block
    bin_half_size = [
        short_side_half_size,
        2 * short_side_half_size + inner_side_half_len,
        2 * short_side_half_size + inner_side_half_len,
    ]  # The bottom block of the bin, which is larger: The list represents the half length of the block along the [x, y, z] axis respectively.
    edge_bin_half_size = [
        short_side_half_size,
        2 * short_side_half_size + inner_side_half_len,
        2 * short_side_half_size,
    ]  # The edge block of the bin, which is smaller. The representations are similar to the above one


    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        assert "camera_width" in kwargs, "camera_width must be provided"
        assert "camera_height" in kwargs, "camera_height must be provided"
        self._camera_width = kwargs.pop("camera_width")
        self._camera_height = kwargs.pop("camera_height")
        self._distraction_set: Union[DistractionSet, dict] = kwargs.pop("distraction_set")
        # In this situation, the DistractionSet has serialized as a dict so we now need to deserialize it.
        if isinstance(self._distraction_set, dict):
            self._distraction_set = DistractionSet(**self._distraction_set)
        self._human_render_shader = kwargs.pop("human_render_shader", None)
        # Env configuration
        self._cube_half_size = 0.02

        """ Coordinate system:

        ===========================
        |                         |
        |                         |
        |    +x ----      --
        |          |        |---()  <- franka-base
        |          |      --
        |          +y       ^franka-gripper
        |                         |
        |                         |
        ===========================
        """

        self._goal_site_cfg = {
            "radius": self._cube_half_size + 0.03,
            "color": [0, 1, 0, 0.75],
            "pose": sapien.Pose(p=[-0.1, 0.35, 0.05]),
        }
        self._cube_cfg = {
            "color": [1, 0, 0, 1],
            "x_bounds": (0.0, -0.1),
            "y_bounds": (-0.35, -0.45),
        }


        y_offset = 0.15
        self._cylinder_obstacle_cfg = [
            {
                "radius": 0.025,
                "height": 0.2,
                "color": [0, 1, 1, 1.0],
                "pose": sapien.Pose(p=[-0.1, y_offset, 0.1], q=[0, 0.7071068, 0, 0.7071068]),
            },{
                "radius": 0.025,
                "height": 0.25,
                "color": [0, 1, 1, 1.0],
                "pose": sapien.Pose(p=[0.0, y_offset, 0.125], q=[0, 0.7071068, 0, 0.7071068]),
            },{
                "radius": 0.025,
                "height": 0.2,
                "color": [0, 1, 1, 1.0],
                "pose": sapien.Pose(p=[0.1, y_offset, 0.1], q=[0, 0.7071068, 0, 0.7071068]),
            },
        ]

        # Note(@jstmn): For some bizzaire reason, you need to create the array with the correct size first,
        # otherwise collecting demonstrations uses an increasing amount of cuda memory and is also much slower. This 
        # took a whilte to debug.
        self._obstacles: List[Optional[sapien.Actor]] = [None] * len(self._cylinder_obstacle_cfg)
        self.goal_site: Optional[sapien.Actor] = None
        self.bin: Optional[sapien.Actor] = None
        self.cube: Optional[sapien.Actor] = None
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)


    def _build_bin(self, radius):
        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose()

        # init the locations of the basic blocks
        dx = self.bin_half_size[1] - self.bin_half_size[0]
        dy = self.bin_half_size[1] - self.bin_half_size[0]
        dz = self.edge_bin_half_size[2] + self.bin_half_size[0]

        # build the bin bottom and edge blocks
        poses = [
            sapien.Pose([0, 0, 0]),
            sapien.Pose([-dx, 0, dz]),
            sapien.Pose([dx, 0, dz]),
            sapien.Pose([0, -dy, dz]),
            sapien.Pose([0, dy, dz]),
        ]
        half_sizes = [
            [self.bin_half_size[1], self.bin_half_size[2], self.bin_half_size[0]],
            self.edge_bin_half_size,
            self.edge_bin_half_size,
            [
                self.edge_bin_half_size[1],
                self.edge_bin_half_size[0],
                self.edge_bin_half_size[2],
            ],
            [
                self.edge_bin_half_size[1],
                self.edge_bin_half_size[0],
                self.edge_bin_half_size[2],
            ],
        ]
        for pose, half_size in zip(poses, half_sizes):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size)

        # build the kinematic bin
        return builder.build_kinematic(name="bin")


    def _get_obs_extra(self, info: dict):
        obs = {}
        def entry_name(link_name, obj_name):
            return f"contact-force:{link_name}___{obj_name}"

        for link_name, link in self.agent.robot.links_map.items():
            for obj_name, obj in zip(["cube", "goal_site", "bin"], [self.cube, self.goal_site, self.bin]):
                forces = self.scene.get_pairwise_contact_forces(link, obj)
                obs[entry_name(link_name, obj_name)] = forces
            for i, obstacle in enumerate(self._obstacles):
                if obstacle is None:
                    continue
                forces = self.scene.get_pairwise_contact_forces(link, obstacle)
                obs[entry_name(link_name, f"cylinder{i}")] = forces
        return obs



    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            x_range = self._cube_cfg["x_bounds"][1] - self._cube_cfg["x_bounds"][0]
            y_range = self._cube_cfg["y_bounds"][1] - self._cube_cfg["y_bounds"][0]
            xyz[:, 0] = torch.rand((b)) * x_range + self._cube_cfg["x_bounds"][0]
            xyz[:, 1] = torch.rand((b)) * y_range + self._cube_cfg["y_bounds"][0]
            xyz[:, 2] = self._cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            # Fixed target position
            self.goal_site.set_pose(self._goal_site_cfg["pose"])
            #
            # for i, cfg in enumerate(self._obstacle_cfgs):
            #     self._obstacles[i].set_pose(cfg["pose"])

            # 
            bin_pos = torch.zeros((b, 3))
            bin_pos[:, 0] = self._goal_site_cfg["pose"].p[0].item()
            bin_pos[:, 1] = self._goal_site_cfg["pose"].p[1].item()
            bin_pos[:, 2] = self.bin_half_size[0]  # on the table
            bin_pose = Pose.create_from_pq(p=bin_pos, q=[1, 0, 0, 0])
            self.bin.set_pose(bin_pose)



    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self._cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self._cube_half_size]),
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self._goal_site_cfg["radius"],
            color=self._goal_site_cfg["color"],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

        # 
        for i, cfg in enumerate(self._cylinder_obstacle_cfg):
            self._obstacles[i] = actors.build_cylinder(
                self.scene,
                radius=cfg["radius"],
                half_length=cfg["height"]/2.0,
                color=cfg["color"],
                name=f"obstacle_{i}",
                body_type="dynamic",
                add_collision=True,
                scene_idxs=None,
                initial_pose=cfg["pose"],
            )
            # self._obstacles[i] = actors.build_box(
            #     self.scene,
            #     half_sizes=cfg["half_size"],
            #     color=cfg["color"],
            #     name=f"obstacle_{i}",
            #     initial_pose=cfg["pose"],
            # )

        self.bin = self._build_bin(self._cube_half_size)


        # Add a ycb object - not important but will be useful later
        builder = get_ycb_builder(self.scene, id="011_banana")
        builder.initial_pose = sapien.Pose(p=[0.35, 0.35, 0.0])
        builder.set_scene_idxs([0])
        self._ycb_obj = builder.build(name="011_banana")
        # self._ycb_obj.set_pose(sapien.Pose(p=[0.35, 0.35, 0.0]))


    @property
    def _default_human_render_camera_configs(self):
        return get_human_render_camera_config(eye=[0.5, 0.6, 0.7], target=[-0.1, 0.0, 0.1], shader=self._human_render_shader)

    @property
    def _default_sensor_configs(self):
        target = [-0.1, 0, 0.0]
        eye_xy = 0.75
        eye_z = 0.75
        cfgs = get_camera_configs(eye_xy, eye_z, target, self._camera_width, self._camera_height)
        cfgs_adjusted = self._distraction_set.update_camera_configs(cfgs)
        return cfgs_adjusted