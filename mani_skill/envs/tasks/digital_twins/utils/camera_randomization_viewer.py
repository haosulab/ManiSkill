"""Creates a viewable uniform pointcloud of a user defined partial spherical shell, representing camera positions for RGB viewpoint randomization"""
from dataclasses import dataclass

import numpy as np
import sapien
import torch
import tyro
from camera_randomization import make_camera_partial_spherical_shell

from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.actors.common import build_sphere
from mani_skill.utils.registration import *


@dataclass
class Args:
    env_id: str = ""
    """the environment id"""

    # pointcloud parameters
    n_points: int = 1000
    """seed of the experiment"""
    r1: float = 1
    """inner_spherical shell radius"""
    r2: float = 1
    """inner_spherical shell radius"""
    max_height: float = 1
    """maximum z value of the pointcloud"""
    max_theta: float = np.pi
    """azimuthal angle describing horizontal width of partial spherical shell"""
    z_orientation: float = 0
    """rotation of pointcloud about z axis"""
    x: float = 0
    """x component center of spherical shell"""
    y: float = 0
    """y component center of spherical shell"""
    z: float = 0
    """z component center of spherical shell"""

    # sample camera parameters for sapien camera sensor viewing
    n_cameras: int = 0
    """number of camera views to sample among the viewpoints"""
    target_x: float = -0.2
    """target x component of the lookat transformation"""
    target_y: float = 0
    """target y component of the lookat transformation"""
    target_z: float = 0.1
    """target y component of the lookat transformation"""
    resolution_width: int = 128
    """camera resolution width"""
    resolution_height: int = 128
    """camera resolution height"""
    fov: float = 1
    """camera fov parameter"""
    near: float = 0.01
    """camera near plane parameter"""
    far: float = 100
    """camera far plane parameter"""


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert (
        0 < args.max_height <= 1
    ), "samples on unit hemisphere slice must have z value between 0 and 1"
    assert (
        0 < args.max_theta <= np.pi
    ), "samples on unit hemisphere slice must have theta value between 0 and pi"
    assert 0 < args.r1 < args.r2
    assert args.n_points > 0
    assert args.max_height <= args.r2
    center = [args.x, args.y, args.z]
    points = make_camera_partial_spherical_shell(
        args.n_points,
        center,
        args.r1,
        args.r2,
        args.max_height,
        args.max_theta,
        args.z_orientation,
    )

    env_class = REGISTERED_ENVS[args.env_id].cls

    @register_env("CopyEnv-v1", max_episode_steps=1e6)
    class CopyEnv(env_class):
        def _load_scene(self, options: dict):
            super()._load_scene(options)
            self.camera_positions = []
            for i in range(len(points)):
                sphere = build_sphere(
                    self.scene,
                    radius=5e-3,
                    color=[1, 0, 0, 1],
                    name=f"c{i}_pos",
                    body_type="static",
                    add_collision=False,
                    initial_pose=sapien.Pose(p=points[i]),
                )
                self.camera_positions.append(sphere)

        @property
        def _default_sensor_configs(self):
            cameras = []
            for i in range(args.n_cameras):
                pt_index = np.random.choice(args.n_points)
                pose = sapien_utils.look_at(
                    eye=list(points[pt_index]),
                    target=[args.target_x, args.target_y, args.target_z],
                )
                camera = CameraConfig(
                    f"camera{pt_index}",
                    pose=pose,
                    width=args.resolution_width,
                    height=args.resolution_height,
                    fov=1,
                    near=0.01,
                    far=100,
                )
                cameras.append(camera)
            return cameras

    env = gym.make("CopyEnv-v1", num_envs=1, render_mode="human")
    env.reset()
    while True:
        env.render()
