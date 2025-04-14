import numpy as np
from typing import Union

import torch
import sapien

from mani_skill.envs.tasks.tabletop.pick_cube_v3 import PickCubeV3Env
from mani_skill.utils.building import actors
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.registration import register_env

@register_env("PickCube-v3-VisibleSphere", max_episode_steps=100)
class PickCubeV3VisibleSphereEnv(PickCubeV3Env):

    """
    **Task Description:**
    Copy of PickCubeEnvV3, but includes a small sphere to visualize the goal position.
    """
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):

        # Configuration for the visible goal sites
        self._visible_goal_site_radius = 0.01
        self._visible_goal_site_color = [0, 0, 1, 1.0]
        # Pointcloud bounds: [-0.175, -0.175, 0.01], [0.175, 0.175, 0.35]
        self._visible_goal_offsets = [
            [0.1, 0.1,  0.0],  # 5 (goal center) + 2 (visible goal site diameter) + 10 (offset) = 17
            [0.1, 0.0,    0.0],
            # [0.0,   0.1,  0.0],
        ]
        self._visible_goal_sites = [None] * len(self._visible_goal_offsets)
        # Note(@jstmn): For some bizzaire reason, you need to create the array with the correct size first,
        # otherwise collecting demonstrations uses an increasing amount of cuda memory before failing. This took too 
        # long to debug.
        self._n_visible_goal_sites = len(self._visible_goal_offsets)

        # Initialize
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)
        print(" --> Created PickCubeV3-VisibleSphere")


    def _load_scene(self, options: dict):
        """ Load the scene.
        """
        super()._load_scene(options)

        # Create goal site
        goal_site = np.array([0.05, 0.05, 0.25])
        for i in range(self._n_visible_goal_sites):
            pose_i = sapien.Pose(p=goal_site + self._visible_goal_offsets[i])
            self._visible_goal_sites[i] = actors.build_sphere(
                self.scene,
                radius=self._visible_goal_site_radius,
                color=self._visible_goal_site_color,
                name=f"visible_goal_site_{i}",
                body_type="static",
                add_collision=True,
                initial_pose=pose_i,
            )
