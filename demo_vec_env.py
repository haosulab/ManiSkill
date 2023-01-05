import argparse
import time
from functools import partial

import gym
import numpy as np

from mani_skill2.utils.visualization.cv2_utils import OpenCVViewer
from mani_skill2.utils.visualization.misc import observations_to_images, tile_images
from mani_skill2.vector import VecEnv, make


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env-id", type=str, default="PickCube-v0")
    parser.add_argument("-o", "--obs-mode", type=str, default="pointcloud")
    parser.add_argument("--reward-mode", type=str)
    parser.add_argument("-c", "--control-mode", type=str)
    parser.add_argument("-n", "--n-envs", type=int, default=4)
    parser.add_argument("--vis", action="store_true")
    args, opts = parser.parse_known_args()

    # Parse env kwargs
    print("opts:", opts)
    eval_str = lambda x: eval(x[1:]) if x.startswith("@") else x
    env_kwargs = dict((x, eval_str(y)) for x, y in zip(opts[0::2], opts[1::2]))
    print("env_kwargs:", env_kwargs)
    args.env_kwargs = env_kwargs

    return args


def main():
    np.set_printoptions(suppress=True, precision=3)
    args = parse_args()

    env_fns = []
    for pid in range(args.n_envs):
        env_fn = partial(
            gym.make,
            id=args.env_id,
            obs_mode=args.obs_mode,
            reward_mode=args.reward_mode,
            control_mode=args.control_mode,
            **args.env_kwargs,
        )
        env_fns.append(env_fn)

    # env = VecEnv(env_fns, server_address="localhost:15003")
    env = make(
        args.env_id,
        args.n_envs,
        server_address="auto",
        obs_mode=args.obs_mode,
        reward_mode=args.reward_mode,
        control_mode=args.control_mode,
        **args.env_kwargs,
    )
    print("env", env)
    print("Observation space", env.observation_space)
    print("Action space", env.action_space)

    n_ep = 10
    l_ep = 50

    if args.vis:
        viewer = OpenCVViewer()

    tic = time.time()
    for i in range(n_ep):
        print("Episode", i)
        env.reset()
        for t in range(l_ep):
            action = [env.action_space.sample() for _ in range(args.n_envs)]
            obs, reward, info, done = env.step(action)
            # print(t, reward, info, done)

            # Visualize
            if args.vis:
                if env.obs_mode in ["image", "rgbd"]:
                    images = []
                    for cam_images in obs["image"].values():
                        images.extend(
                            observations_to_images(
                                {k: v[0].cpu().numpy() for k, v in cam_images.items()}
                            )
                        )
                    viewer.imshow(tile_images(images))
                elif env.obs_mode == "pointcloud":
                    import trimesh

                    scene = trimesh.Scene()
                    for i_env in range(args.n_envs):
                        xyz = obs["pointcloud"]["xyzw"][i_env, ..., :3].cpu().numpy()
                        rgb = obs["pointcloud"]["rgb"][i_env].cpu().numpy()
                        # trimesh.PointCloud(xyz, rgb).show()
                        T = np.eye(4)
                        T[2, 3] = i_env * 1.0
                        scene.add_geometry(trimesh.PointCloud(xyz, rgb), transform=T)
                    scene.show()

    toc = time.time()
    print("FPS", n_ep * l_ep * args.n_envs / (toc - tic))
    env.close()


if __name__ == "__main__":
    main()
