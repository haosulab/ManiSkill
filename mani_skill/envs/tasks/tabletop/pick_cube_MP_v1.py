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
from mani_skill.envs.tasks.tabletop.get_camera_config import get_camera_configs


@register_env("PickCubeMP-v1", max_episode_steps=100)
class PickCubeMPEnv(PickCubeEnv):
    """
    **Task Description:**
    Nearly exacty copy of PickCubeEnv, but with the following change:
        1. 3 cameras instead of 1
        2. Cameras have a higher resolution
        3. Target position is fixed to (0.05, 0.05, 0.25)
        4. Goal_thresh is the cube half size plus a small margin

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
        self._camera_width = kwargs.pop("camera_width")
        self._camera_height = kwargs.pop("camera_height")

        # Env configuration
        self._cube_half_size = 0.02

        self._goal_site_cfg = {
            "radius": self._cube_half_size + 0.05,
            "color": [0, 1, 0, 0.75],
            "pose": sapien.Pose(p=[0.0, 0.4, 0.25]),
        }
        self._cube_cfg = {
            "color": [1, 0, 0, 1],
            "x_bounds": (-0.1, 0.1),
            "y_bounds": (-0.3, -0.4),
        }

        self._obstacle_cfgs = [
            {
                "half_size": [0.05, 0.05, 0.15],
                "color": [0, 1, 1, 1.0],
                "pose": sapien.Pose(p=[-0.1, 0, 0.05]),
            },
            {
                "half_size": [0.05, 0.05, 0.1],
                "color": [0, 1, 1, 1.0],
                "pose": sapien.Pose(p=[0.1, 0, 0.05]),
            }
        ]
        self._obstacles = []
        super().__init__(*args, robot_uids=robot_uids, robot_init_qpos_noise=robot_init_qpos_noise, **kwargs)


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
            print("xyz:", xyz)
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            # Fixed target position
            self.goal_site.set_pose(self._goal_site_cfg["pose"])
            #
            # for i, cfg in enumerate(self._obstacle_cfgs):
            #     self._obstacles[i].set_pose(cfg["pose"])


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
        for i, cfg in enumerate(self._obstacle_cfgs):
            self._obstacles.append(actors.build_box(
                self.scene,
                half_sizes=cfg["half_size"],
                color=cfg["color"],
                name=f"obstacle_{i}",
                initial_pose=cfg["pose"],
            ))


    @property
    def _default_human_render_camera_configs(self):
        """ Configures the human render camera.

        Shader options:
            minimal: The fastest shader with minimal GPU memory usage. Note that the background will always be black (normally it is the color of the ambient light)
            default: A balance between speed and texture availability
            rt: A shader optimized for photo-realistic rendering via ray-tracing
            rt-med: Same as rt but runs faster with slightly lower quality
            rt-fast: Same as rt-med but runs faster with slightly lower quality

            -> https://maniskill.readthedocs.io/en/latest/user_guide/concepts/sensors.html#shaders-and-textures
        """

        pose = sapien_utils.look_at([0.35, 0.45, 0.4], [0.0, 0.0, 0.15])
        # SHADER = "rt" # doesn't work with parallel rendering
        SHADER = "default"
        # SHADER = "minimal" # negligible time difference between 'default' and 'minimal', so 
        return CameraConfig("render_camera", pose=pose, width=1264, height=1264, fov=np.pi / 3, near=0.01, far=100, shader_pack=SHADER)

    @property
    def _default_sensor_configs(self):
        target = [-0.1, 0, 0.1]
        eye_xy = 0.5
        eye_z = 0.6
        return get_camera_configs(eye_xy, eye_z, target, self._camera_width, self._camera_height)