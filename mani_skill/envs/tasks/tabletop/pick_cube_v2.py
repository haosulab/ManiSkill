import numpy as np

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

REALSENSE_DEPTH_FOV_VERTICAL_RAD = 58.0 * np.pi / 180
REALSENSE_DEPTH_FOV_HORIZONTAL_RAD = 87.0 * np.pi / 180


@register_env("PickCube-v2", max_episode_steps=100)
class PickCubeV2Env(PickCubeEnv):
    """
    **Task Description:**
    Nearly exacty copy of PickCubeEnv, but with the following change:
        1. 3 cameras instead of 1
        2. Cameras have a higher resolution
        3. Target position is fixed to (0.05, 0.05, 0.25)

    Distractor axes:
        1. Cube visual. In build_cube(...) add args to RenderMaterial:
            builder.add_box_visual(
                half_size=[half_size] * 3,
                material=sapien.render.RenderMaterial(
                    base_color=color,
                ),
            )
            (https://sapien.ucsd.edu/docs/latest/apidoc/sapien.core.html#sapien.core.pysapien.RenderMaterial)
        2.
    """
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        assert "camera_width" in kwargs, "camera_width must be provided"
        assert "camera_height" in kwargs, "camera_height must be provided"
        # assert "distractor_specification" in kwargs, "distractor_specification must be provided"
        self._camera_width = kwargs.pop("camera_width")
        self._camera_height = kwargs.pop("camera_height")
        # self._distractor_specification = kwargs.pop("distractor_specification")

        self._distractor_spheres = []
        self._n_distractor_spheres = 0
        self._distractor_spheres_radius_range = (0.01, 0.025)
        self._distractor_spheres_color_range = [[0.25, 0.25, 0.25], [1.0, 1.0, 1.0]]

        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1

            # TEMP DEBUGGING - Fixed cube start position
            # xyz[:, 0] = 0.05
            # xyz[:, 1] = 0.05
            # END OF: TEMP DEBUGGING

            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            # Fixed target position
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, 0] = 0.05
            goal_xyz[:, 1] = 0.05
            goal_xyz[:, 2] = 0.25
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

            # New distractor spheres
            for i in range(self._n_distractor_spheres):
                sphere_xyz = torch.zeros((b, 3))
                sphere_xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
                sphere_xyz[:, 2] = 0.05
                self._distractor_spheres[i].set_pose(Pose.create_from_pq(sphere_xyz))


    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

        # New distractor spheres
        for i in range(self._n_distractor_spheres):
            radius_i = np.random.uniform(*self._distractor_spheres_radius_range)
            self._distractor_spheres.append(actors.build_sphere(
                self.scene,
                initial_pose=sapien.Pose(p=[0.1, 0.1, radius_i]),
                name=f"distractor_sphere_{i}",
                radius=radius_i,
                color=np.random.uniform(*self._distractor_spheres_color_range).tolist() + [1.0], # alpha=1.0
            ))


    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.3, 0.4, 0.4], [0.0, 0.0, 0.25])
        # SHADER = "rt" # doesn't work with parallel rendering
        SHADER = "default"
        print(f"Using shader '{SHADER}' for human render camera")
        return CameraConfig("render_camera", pose=pose, width=1264, height=1264, fov=np.pi / 2, near=0.01, far=100, shader_pack=SHADER)

    @property
    def _default_sensor_configs(self):
        pose_center = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        pose_left = sapien_utils.look_at(eye=[0.0, -0.3, 0.6], target=[-0.1, 0, 0.1])
        pose_right = sapien_utils.look_at(eye=[0.0, 0.3, 0.6], target=[-0.1, 0, 0.1])
        SHADER = "default"
        print(f"Using shader '{SHADER}' for sensor cameras")
        return [
            CameraConfig(
                uid="camera_center",
                pose=pose_center,
                width=self._camera_width,
                height=self._camera_height,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                shader_pack=SHADER,
            ),
            CameraConfig(
                uid="camera_left",
                pose=pose_left,
                width=self._camera_width,
                height=self._camera_height,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                shader_pack=SHADER,
            ),
            CameraConfig(
                uid="camera_right",
                pose=pose_right,
                width=self._camera_width,
                height=self._camera_height,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                shader_pack=SHADER,
            ),
        ]