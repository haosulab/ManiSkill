import gymnasium as gym
import numpy as np
import pytest
import sapien
import sapien.physx
import torch

from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.sapien_utils import to_numpy

torch.set_printoptions(precision=3, sci_mode=False)


def test_parallel_actors():
    if not sapien.physx.is_gpu_enabled():
        sapien.physx.enable_gpu()
    sapien.set_cuda_tensor_backend("torch")

    def create_scene(offset):
        scene = sapien.Scene()
        scene.physx_system.set_scene_offset(scene, offset)
        scene.set_ambient_light([0.5, 0.5, 0.5])
        return scene

    num_scenes = 4
    sub_scenes = []
    for i in range(num_scenes):
        sub_scenes.append(create_scene([i * 10, 0, 0]))

    from mani_skill2.envs.scene import ManiSkillScene

    scene = ManiSkillScene(sub_scenes, debug_mode=True)
    # scene.sub_scenes = [scene0, scene1]

    builder = scene.create_actor_builder()
    builder.add_box_visual(half_size=(0.25, 0.25, 0.25), material=[1, 0, 0, 1])
    builder.add_box_collision(half_size=(0.25, 0.25, 0.25))
    builder.initial_pose = sapien.Pose(p=[2, 0, 1])
    builder.build(name="cube-0")

    builder = scene.create_actor_builder()
    builder.add_box_visual(half_size=(0.25, 0.25, 0.25), material=[1, 0, 0, 1])
    builder.add_box_collision(half_size=(0.25, 0.25, 0.25))
    builder.initial_pose = sapien.Pose(p=[5, 5, 5])
    builder.build(name="cube-1")

    scene._setup_gpu()

    raw_pose = to_numpy(scene.actors["cube-0"].pose.raw_pose)
    assert np.isclose(
        raw_pose[0], np.array([2, 0, 1, 1, 0, 0, 0])
    ).all(), "pose in scene 0 is wrong"
    assert np.isclose(
        raw_pose, raw_pose[0]
    ).all(), "poses are not synced across sub scenes"
    raw_pose = to_numpy(scene.actors["cube-1"].pose.raw_pose)
    assert np.isclose(
        raw_pose[0], np.array([5, 5, 5, 1, 0, 0, 0])
    ).all(), "pose in scene 0 is wrong"
    assert np.isclose(
        raw_pose, raw_pose[0]
    ).all(), "poses are not synced across sub scenes"

    scene.px.step()
    scene._gpu_fetch_all()
    assert np.isclose(
        to_numpy(scene.actors["cube-1"].get_raw_data()["linear_vel"]),
        np.array([0, 0, -0.0981]),
    ).all(), "linear velocity in scene 0 after one sim step is wrong"


def test_env_step():
    num_envs = 4
    env: BaseEnv = gym.make(
        "PickCube-v0",
        num_envs=num_envs,
        render_mode="human",
        control_mode="pd_joint_pos",
        sim_freq=200,
    )
    env.reset(seed=0)
    env = env.unwrapped
    scene = env._scene
    # cube = scene.actors["cube"]
    # print(cube.get_raw_data())
    table = scene.actors["table-workspace"]
    # expect kinematic actors to not move
    orig_table_pose = table.pose.raw_pose.cpu().numpy()
    for _ in range(2):
        env.step(None)
    assert np.isclose(orig_table_pose, table.pose.raw_pose.cpu().numpy()).all()
