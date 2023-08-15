import argparse
import time

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from mani_skill2.utils.visualization.misc import observations_to_images, tile_images
from mani_skill2.vector import VecEnv, make


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--env-id",
        type=str,
        default="PickCube-v0",
        help="The environment to this demo on",
    )
    parser.add_argument(
        "-o",
        "--obs-mode",
        type=str,
        default="image",
        help="The observation mode to use",
    )
    parser.add_argument(
        "-c", "--control-mode", type=str, help="The control mode to use"
    )
    parser.add_argument("--reward-mode", type=str, help="The reward mode to use")
    parser.add_argument(
        "-n",
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments to run",
    )
    parser.add_argument(
        "--vis", action="store_true", help="Whether to visualize the environments"
    )
    parser.add_argument(
        "--n-ep",
        type=int,
        default=5,
        help="Number of episodes to run per parallel environment",
    )
    parser.add_argument(
        "--l-ep", type=int, default=200, help="Max number of timesteps per episode"
    )
    parser.add_argument(
        "--vecenv-type",
        type=str,
        default="ms2",
        help="Type of VecEnv to use. Can be ms2 or gym",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output.")
    args, opts = parser.parse_known_args(args)

    # Parse env kwargs
    if not args.quiet:
        print("opts:", opts)
    eval_str = lambda x: eval(x[1:]) if x.startswith("@") else x
    env_kwargs = dict((x, eval_str(y)) for x, y in zip(opts[0::2], opts[1::2]))
    if not args.quiet:
        print("env_kwargs:", env_kwargs)
    args.env_kwargs = env_kwargs

    return args


def main(args):
    np.set_printoptions(suppress=True, precision=3)

    verbose = not args.quiet
    n_ep = args.n_ep
    l_ep = args.l_ep

    if args.vecenv_type == "ms2":
        env: VecEnv = make(
            args.env_id,
            args.n_envs,
            obs_mode=args.obs_mode,
            reward_mode=args.reward_mode,
            control_mode=args.control_mode,
            **args.env_kwargs,
        )
    elif args.vecenv_type == "gym":
        env = gym.make_vec(
            args.env_id,
            args.n_envs,
            vectorization_mode="async",
            reward_mode=args.reward_mode,
            obs_mode=args.obs_mode,
            control_mode=args.control_mode,
            vector_kwargs=dict(context="forkserver"),
        )
    else:
        raise ValueError(f"{args.vecenv_type} is invalid. Must be ms2 or gym")
    if verbose:
        print(f"Environment {args.env_id} - {args.n_envs} parallel envs")
        print("Observation space", env.observation_space)
        print("Action space", env.action_space)

    np.random.seed(2022)

    samples_so_far = 0
    total_samples = n_ep * l_ep * args.n_envs
    tic = time.time()

    pbar = tqdm(range(n_ep))
    for i in pbar:
        # NOTE(jigu): reset is a costly operation
        obs, _ = env.reset()

        for t in range(l_ep):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            samples_so_far += args.n_envs

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
        fps = samples_so_far / (time.time() - tic)
        pbar.set_postfix(dict(FPS=f"{fps:0.2f}"))
    toc = time.time()
    if verbose:
        print(f"FPS {total_samples / (toc - tic):0.2f}")
    env.close()


if __name__ == "__main__":
    main(parse_args())
