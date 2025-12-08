import gymnasium as gym
import numpy as np
import time
from sapien.utils import Viewer
from tests.utils import ENV_IDS # type: ignore
from sapien import Pose


# TODO: This should be a manual only test.
def test_viewer():
    """ Tests that we can add the following shapes to the viewer:
    - bounding box
    - AABB
    - 3D point list
    - coordinate frame
    """
    env = gym.make("PushCube-v1", render_mode="human")
    env.reset()
    viewer = env.render()
    assert isinstance(viewer, Viewer)    
    
    # Test 1: add_bounding_box()
    box = viewer.add_bounding_box(
        Pose(p=np.array([0, 0, 0]), q=np.array([0, 0, 0, 1])), 
        half_size=np.array([1, 1, 1]), 
        color=np.array([1, 0, 0, 1])
    )

    # Test 2: add_AABB()
    aabb = viewer.draw_aabb(lower=np.array([0, 0, 0]), upper=np.array([0.5, 0.5, 0.5]), color=np.array([0, 1, 0, 1]))

    # Test 3.a: add_3D_point_list() 
    points_x = np.linspace(0, 1, 50)
    points_y = 0.1*np.sin(points_x * 2 * np.pi)
    points_z = 1.0 + 0.1*np.cos(points_x * 2 * np.pi)
    points = np.column_stack([points_x, points_y, points_z])
    pointset_1 = viewer.add_3d_point_list(points=points, color=np.array([0, 0, 1, 1]))

    # Test 3.b: add_3D_point_list() with per-point colors
    points_2 = points.copy()
    points_2[:, 2] -= 0.25
    colors = np.zeros((50, 3))
    colors[:, 0] = np.sin(np.linspace(0, 10, 50))
    colors[:, 1] = np.sin(np.linspace(2, 10, 50))
    colors[:, 2] = np.sin(np.linspace(4, 10, 50))
    pointset_2 = viewer.add_3d_point_list(points=points_2, color=colors)


    # Test 4.a: add_coordinate_frame() 
    pose_1 = Pose(p=np.array([0.25, 0.25, 0.25]), q=np.array([1, 0, 0, 0]))
    frame_1 = viewer.add_coordinate_frame(pose=pose_1)
    
    pose_2 = Pose(p=np.array([0.75, 0.25, 0.25]), q=np.array([0.7071068, 0, 0, 0.7071068]))
    frame_2 = viewer.add_coordinate_frame(pose=pose_2)
    

    env.render()
    viewer.paused = True

    for i in range(10):
        time.sleep(0.1)
        env.render()
    
    viewer.paused = True
    viewer.remove_bounding_box(box)
    viewer.remove_aabb(aabb)
    viewer.remove_3d_point_list(pointset_1)
    viewer.remove_3d_point_list(pointset_2)
    viewer.remove_coordinate_frame(frame_1)
    viewer.remove_coordinate_frame(frame_2)
    env.render()
    viewer.paused = True
    env.render()
    env.close()

