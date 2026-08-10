import time

import gymnasium as gym
import numpy as np
import pytest
import sapien

import mani_skill.envs  # noqa: F401

RUNTIME_ALLOWED = 10

@pytest.mark.slow
def test_viser_debug_visualization():
    """Load PickCube with the Viser visualizer, draw some debug items, then shut down."""
    env = gym.make(
        "PickCube-v1",
        obs_mode="state",
        render_mode="rgb_array",
        visualizer_backend="viser",
        sim_backend="cpu",
        num_envs=1,
    )
    env.reset(seed=0)
    base_env = env.unwrapped
    vv = base_env.scene.viser_visualizer
    assert vv is not None

    rng = np.random.default_rng(0)

    # Random bounding boxes
    boxes = []
    for _ in range(3):
        pose = sapien.Pose(p=rng.uniform(-0.2, 0.2, size=3) + np.array([0.0, 0.0, 0.15]))
        half_size = rng.uniform(0.02, 0.06, size=3)
        color = rng.random(4)
        color[3] = 1.0
        boxes.append(vv.add_bounding_box(pose, half_size, color, line_width=2.0))

    # Random AABBs
    aabbs = []
    for _ in range(2):
        center = rng.uniform(-0.15, 0.15, size=3) + np.array([0.0, 0.0, 0.2])
        half = rng.uniform(0.02, 0.05, size=3)
        aabbs.append(
            vv.draw_aabb(center - half, center + half, rng.random(3), line_width=1.5)
        )

    # Random point clouds
    point_sets = []
    for _ in range(2):
        points = rng.normal(size=(40, 3)).astype(np.float32) * 0.05
        points[:, 2] += 0.25
        colors = rng.random((40, 3))
        point_sets.append(vv.add_3d_point_list(points, color=colors))

    # Random coordinate frames
    frames = []
    for _ in range(3):
        pose = sapien.Pose(p=rng.uniform(-0.2, 0.2, size=3) + np.array([0.0, 0.0, 0.1]))
        frames.append(vv.add_coordinate_frame(pose, length=0.08, radius=0.004))

    # Keep the sim/visualizer alive for 40s so the browser view can be inspected
    deadline = time.time() + RUNTIME_ALLOWED
    while time.time() < deadline:
        action = env.action_space.sample() if env.action_space is not None else None
        env.step(action)
        time.sleep(0.05)

    for box in boxes:
        vv.remove_bounding_box(box)
    for aabb in aabbs:
        vv.remove_aabb(aabb)
    for point_set in point_sets:
        vv.remove_3d_point_list(point_set)
    for frame in frames:
        vv.remove_coordinate_frame(frame)

    env.close()
