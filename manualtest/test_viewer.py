import time
import gymnasium as gym
import sapien
import mani_skill2.envs
from mani_skill2.envs.sapien_env import BaseEnv
import sapien.physx
import tqdm
import torch
import sapien.render
import sapien.physx
import matplotlib.pyplot as plt
from mani_skill2.envs.scene import ManiSkillScene
from mani_skill2.utils.visualization.misc import images_to_video
from mani_skill2.utils.wrappers.observation import FlattenObservationWrapper
from mani_skill2.utils.wrappers.record import RecordEpisode
if __name__ == "__main__":
    torch.set_printoptions(precision=3, sci_mode=False)
    num_envs = 2
    env: BaseEnv = gym.make("PickCube-v0", num_envs=num_envs, obs_mode="rgbd", render_mode="human", control_mode="pd_joint_delta_pos", sim_freq=200)
    print("GPU Enabled:", sapien.physx.is_gpu_enabled())
    
    print("= ENV RESET")
    obs, _ = env.reset(seed=0)
    # env=env.unwrapped
    scene=env.unwrapped._scene
    # from sapien.utils import Viewer
    # scene.update_render()

    # viewer = Viewer()
    # rs: sapien.internal_renderer.Scene = scene.sub_scenes[1].render_system._internal_scene
    # rs.set_root_transform([2, 0, 0], [1, 0, 0, 0], [1, 1, 1])

    # # viewer.set_scene(scenes[0])
    # viewer.set_scenes(scene.sub_scenes)
    # vs = viewer.window._internal_scene
    # vs.set_ambient_light([0.3, 0.3, 0.3])
    # # vs.set_cubemap(scene.sub_scenes[0].render_system.get_cubemap()._internal_cubemap)

    # viewer.render()

    # while not viewer.closed:
    #     scene.px.step()
    #     scene.px.sync_poses_gpu_to_cpu()
    #     viewer.window.update_render()
    #     viewer.render()
    # table = scene.actors["table-workspace"]
    # cube=env.obj
    # import ipdb;ipdb.set_trace()
    # viewer = env.render()
    # viewer.paused=True
    # view_scene_idx = 1
    # env._viewer_scene_idx = view_scene_idx
    # viewer.set_scene(scene.sub_scenes[view_scene_idx])
    # while True:
    #     window = viewer.window
    #     if window.key_press("n"):
    #         view_scene_idx = (view_scene_idx + 1) % len(scene.sub_scenes)
    #         env._viewer_scene_idx = view_scene_idx
    #         viewer.set_scene(scene.sub_scenes[view_scene_idx])
    #     env.render()
    #     env.step(env.action_space.sample())
    
    # render_system_group = scene.render_system_group
    # scene._gpu_setup_sensors(env._sensors)
    # scene._gpu_setup_sensors(env._render_cameras)
    scene: ManiSkillScene
    # scene.update_render()

    # scene.camera_groups["base_camera"].take_picture()
    # picture = scene.camera_groups["base_camera"].get_picture_cuda("Color")
    # picture = picture.cpu().numpy()
    images = []
    for _ in range(25):
        obs, rew, terminated, truncated, info = env.step(env.action_space.sample()*0)
        print(info, rew)
        # scene.update_render()
        # # scene.camera_groups["render_camera"].take_picture()
        # scene.camera_groups["base_camera"].take_picture()
        # picture = scene.camera_groups["base_camera"].get_picture_cuda("Color")
        # picture = env.render_rgb_array(camera_name="render_camera")
        # import ipdb;ipdb.set_trace()
        picture = obs["image"]["hand_camera"]["rgb"]
        picture = picture.cpu().numpy()
        # 
        plt.subplot(1, 2, 1)
        plt.imshow(picture[0])
        plt.subplot(1, 2, 2)
        plt.imshow(picture[1])
        plt.show()