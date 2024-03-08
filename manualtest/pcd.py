# import gymnasium as gym
# import trimesh

# import mani_skill.envs

# env = gym.make(
#     "ReplicaCAD_SceneManipulation-v1",
#     obs_mode="pointcloud",
#     control_mode="pd_joint_delta_pos",
#     num_envs=2,
# )
# print("Observation space", env.observation_space)
# print("Action space", env.action_space)

# obs, _ = env.reset(seed=0)  # reset with a seed for determinism
# for i in range(1000):
#     # action = env.action_space.sample()  # this is batched now
#     # obs, reward, terminated, truncated, info = env.step(action)
#     # done = terminated | truncated

#     if env.unwrapped.num_envs > 1:
#         pts = obs["pointcloud"]["xyzw"][..., :3]
#         colors = obs["pointcloud"]["rgb"][..., :3]
#         mask = obs["pointcloud"]["xyzw"][..., 3] == 1
#         # colors[~mask] *= 0
#         pcd = trimesh.points.PointCloud(pts[0].cpu().numpy(), colors[0].cpu().numpy())
#         cam2world = obs["sensor_param"]["fetch_head"]["cam2world_gl"].cpu().numpy()[0]
#         print(cam2world)
#     else:
#         cam2world = obs["sensor_param"]["fetch_head"]["cam2world_gl"]
#         pcd = trimesh.points.PointCloud(obs["pointcloud"]["xyzw"][..., :3], obs["pointcloud"]["rgb"][..., :3])
#         print(cam2world)
#     # pcd = trimesh.points.PointCloud(pts[0].cpu().numpy(), obs["pointcloud"]["xyzw"][..., 3].cpu().numpy()[0] == 1)
#     import trimesh.scene

#     camera = trimesh.scene.Camera("base", (1024, 1024), fov=(90, 90))
#     trimesh.Scene([pcd], camera=camera, camera_transform=cam2world).show()
#     # trimesh.Scene([pcd], camera_transform=cam2world).show()
#     import ipdb

#     ipdb.set_trace()
#     print(f"Reward shape {reward.shape}, Done shape {done.shape}")
#     # note at the moment we do not support showing all parallel sub-scenes
#     # at once on a GUI, only during observation generation/video recording
# env.close()


import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import trimesh

import mani_skill.envs

env = gym.make(
    "RotateValveLevel0-v1",
    obs_mode="rgbd",
    control_mode="pd_joint_delta_pos",
    num_envs=1,
)
print("Observation space", env.observation_space)
print("Action space", env.action_space)

obs, _ = env.reset(seed=0)  # reset with a seed for determinism
for i in range(1000):
    # action = env.action_space.sample()  # this is batched now
    # obs, reward, terminated, truncated, info = env.step(action)
    # done = terminated | truncated
    for i, uid in enumerate(["base_camera"]):
        img = obs["sensor_data"][uid]["rgb"][0].cpu().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"rgb{i}.png", img)
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        img = obs["sensor_data"][uid]["depth"][0].cpu().numpy()
        ax.imshow(img)
        plt.savefig(f"depth{i}.png")
    break
    # plt.savefig("rgb.png")
    # plt.imshow(obs["sensor_data"]["fetch_hand"]["depth"][0].cpu().numpy())
    # plt.savefig("depth.png")
    # print(f"Reward shape {reward.shape}, Done shape {done.shape}")
    # note at the moment we do not support showing all parallel sub-scenes
    # at once on a GUI, only during observation generation/video recording
env.close()
