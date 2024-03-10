import argparse

import gymnasium as gym
import numpy as np

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
import trimesh
import trimesh.scene
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PushCube-v1", help="The environment ID of the task you want to simulate")
    parser.add_argument("-o", "--obs-mode", type=str, default="none")
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Seed the random actions and simulator. Default is no seed",
    )
    args = parser.parse_args()
    return args


def main(args):
    np.set_printoptions(suppress=True, precision=3)
    if args.seed is not None:
        np.random.seed(args.seed)
    env: BaseEnv = gym.make(
        args.env_id,
        obs_mode="pointcloud",
        reward_mode="sparse",
    )

    obs, _ = env.reset(seed=args.seed)
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        xyz = obs["pointcloud"]["xyzw"][..., :3]
        colors = obs["pointcloud"]["rgb"]
        pcd = trimesh.points.PointCloud(xyz, colors)


        # view from first camera
        for uid, cfg in env.unwrapped._sensor_cfgs.items():
            if isinstance(cfg, CameraConfig):
                cam2world = obs["sensor_param"][uid]["cam2world_gl"]
                camera = trimesh.scene.Camera(uid, (1024, 1024), fov=(np.rad2deg(cfg.fov), np.rad2deg(cfg.fov)))
            break
        trimesh.Scene([pcd], camera=camera, camera_transform=cam2world).show()
        if terminated or truncated:
            break
    env.close()

if __name__ == "__main__":
    main(parse_args())
