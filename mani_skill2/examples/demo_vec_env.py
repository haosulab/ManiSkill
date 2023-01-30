import argparse
import time

import gym
import numpy as np

from mani_skill2 import make_box_space_readable
from mani_skill2.utils.visualization.misc import observations_to_images, tile_images
from mani_skill2.vector import VecEnv, make


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v0")
    parser.add_argument("-o", "--obs-mode", type=str, default="image")
    parser.add_argument("-c", "--control-mode", type=str)
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-n", "--n-envs", type=int, default=4)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--n-ep", type=int, default=5)
    parser.add_argument("--l-ep", type=int, default=200)
    args, opts = parser.parse_known_args()

    # Parse env kwargs
    print("opts:", opts)
    eval_str = lambda x: eval(x[1:]) if x.startswith("@") else x
    env_kwargs = dict((x, eval_str(y)) for x, y in zip(opts[0::2], opts[1::2]))
    print("env_kwargs:", env_kwargs)
    args.env_kwargs = env_kwargs

    return args


def main():
    make_box_space_readable()
    np.set_printoptions(suppress=True, precision=3)

    args = parse_args()
    n_ep = args.n_ep
    l_ep = args.l_ep

    env: VecEnv = make(
        args.env_id,
        args.n_envs,
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        **args.env_kwargs,
    )
    print("env", env)
    print("Observation space", env.observation_space)
    print("Action space", env.action_space)

    np.random.seed(2022)
    env.seed(2022)

    tic = time.time()
    for i in range(n_ep):
        print("Episode", i)

        # NOTE(jigu): reset is a costly operation
        obs = env.reset()
        # env.reset(indices=[j for j in range(args.n_envs) if j % 2 == i % 2])

        for t in range(l_ep):
            action = [env.action_space.sample() for _ in range(args.n_envs)]
            obs, reward, info, done = env.step(action)
            # print(t, reward, info, done)

            # Visualize
            if args.vis and env.obs_mode in ["image", "rgbd", "rgbd_robot_seg"]:
                import cv2

                images = []
                for i_env in range(args.n_envs):
                    for cam_images in obs["image"].values():
                        images_i = observations_to_images(
                            {k: v[i_env].cpu().numpy() for k, v in cam_images.items()}
                        )
                        images.append(np.concatenate(images_i, axis=0))
                cv2.imshow("vis", tile_images(images)[..., ::-1])
                cv2.waitKey(0)

            if args.vis and env.obs_mode in ["pointcloud", "pointcloud_robot_seg"]:
                import trimesh

                scene = trimesh.Scene()
                for i_env in range(args.n_envs):
                    pcd_obs = obs["pointcloud"]
                    xyz = pcd_obs["xyzw"][i_env, ..., :3].cpu().numpy()
                    w = pcd_obs["xyzw"][i_env, ..., 3].cpu().numpy()
                    rgb = pcd_obs["rgb"][i_env].cpu().numpy()
                    if "robot_seg" in pcd_obs:
                        robot_seg = pcd_obs["robot_seg"][i_env].cpu().numpy()
                        rgb = np.uint8(robot_seg * [11, 61, 127])
                    # trimesh.PointCloud(xyz, rgb).show()
                    # Distribute point clouds in z axis
                    T = np.eye(4)
                    T[2, 3] = i_env * 1.0
                    scene.add_geometry(
                        trimesh.PointCloud(xyz[w > 0], rgb[w > 0]), transform=T
                    )
                scene.show()

    toc = time.time()
    print("FPS", n_ep * l_ep * args.n_envs / (toc - tic))
    env.close()


if __name__ == "__main__":
    main()
