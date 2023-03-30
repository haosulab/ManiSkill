# Observation

See our [colab tutorial](https://colab.research.google.com/github/haosulab/ManiSkill2/blob/main/examples/tutorials/customize_environments.ipynb#scrollTo=NaSQ7CD2sswC) for how to customize cameras.

## Observation mode

**The observation mode defines the observation space.**
All ManiSkill2 environments take the observation mode (`obs_mode`) as one of the input arguments of `__init__`.
In general, the observation is organized as a dictionary (with an observation space of `gym.spaces.Dict`).

There are two raw observations modes: `state_dict` (privileged states) and `image` (raw visual observations without postprocessing). `state` is a flat version of `state_dict`. `rgbd` and `pointcloud` apply post-processing on `image`.

### state_dict

The observation is a dictionary of states. It usually contains privileged information such as object poses. It is not supported for soft-body environments.

- `agent`: robot proprioception
  - `qpos`: [nq], current joint positions. *nq* is the degree of freedom.
  - `qvel`: [nq], current joint velocities
  - `base_pose`: [7], robot position (xyz) and quaternion (wxyz) in the world frame
  - `controller`: controller states depending on the used controller. Usually an empty dict.
- `extra`: a dictionary of task-specific information, e.g., goal position, end-effector position.

### state

It is a flat version of *state_dict*. The observation space is `gym.spaces.Box`.

### image

In addition to `agent` and `extra`, `image` and `camera_param` are introduced.

- image: RGB, depth, and other images taken by cameras
  - `{camera_uid}`:
    - `Color`: [H, W, 4], `np.float32`. RGBA.
    - `Position`: [H, W, 4], `np.float32`. The first 3 dimensions stand for (x, y, z) coordinates in the OpenGL/Blender convension. The unit is meter.
- `camera_param`: camera parameters
  - `cam2world_gl`: [4, 4], transformation from the camera frame to the world frame (OpenGL/Blender convention)
  - `extrinsic_cv`: [4, 4], camera extrinsic (OpenCV convention)
  - `intrinsic_cv`: [3, 3], camera intrinsic (OpenCV convention)

Unless specified otherwise, there are two cameras: *base_camera* (fixed relative to the robot base) and *hand_camera* (mounted on the robot hand). Environments migrated from ManiSkill1 use 3 cameras mounted above the robot: *overhead_camera_{i}*.

### rgbd

We postprocess the raw image observation to obtain RGB-D images.

- `image`: RGB, depth, and other images taken by cameras
  - `{camera_uid}`:
    - `rgb`: [H, W, 3], `np.uint8`. RGB.
    - `depth`: [H, W, 1], `np.float32`. 0 stands for an invalid pixel (beyond the camera far).

### pointcloud

We postprocess the raw image observation to obtain a point cloud in the world frame. `image` is replaced by `pointcloud`.

- `pointcloud`:
  - `xyzw`: [N, 4], point cloud fused from all cameras in the world frame. "xyzw" is a homogeneous representation. `w=0` for infinite points (beyond the camera far), and `w=1` for the rest.
  - `rgb`: [N, 3], corresponding colors of the fused point cloud

Note that the point cloud does not contain more information than RGB-D images unless specified otherwise.

### +robot_seg

`rgbd+robot_seg` or `pointcloud+robot_seg`  can be used to acquire the segmentation mask of robot links. `robot_seg` is appended.

- `pointcloud+robot_seg`:
  - `robot_seg`: [N, 1], a binary mask where 1 for robot and 0 for others.

- `rgbd+robot_seg`:
  - {camera_uid}
  - `robot_seg`: [N, 1], a binary mask where 1 for robot and 0 for others.

## Ground-truth Segmentation

Ground-truth segmentation can be used to generate training data for computer vision, reinforcement learning, and many other applications.

```python
env = gym.make(env_id, camera_cfgs={"add_segmentation": True})
```

There will be an additional key: *Segmentation*.

For `obs_mode="rgbd"`:

- `image`:
  - `{camera_uid}`
    - `Segmentation`: [H, W, 4], `np.uint32`. The 1st dimension is mesh-level (part) segmentation. The 2nd dimension is actor-level (object/link) segmentation.

For `obs_mode="pointcloud"`:

- `pointcloud`:
  - `Segmentation`: [N, 4], `np.uint32`
