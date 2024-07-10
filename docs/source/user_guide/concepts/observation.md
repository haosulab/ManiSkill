# Observation

<!-- See our [colab tutorial](https://colab.research.google.com/github/haosulab/ManiSkill/blob/main/examples/tutorials/customize_environments.ipynb#scrollTo=NaSQ7CD2sswC) for how to customize cameras. -->

## Observation mode

**The observation mode defines the observation space.**
All ManiSkill tasks take the observation mode (`obs_mode`) as one of the input arguments of `__init__`.
In general, the observation is organized as a dictionary (with an observation space of `gym.spaces.Dict`).

There are two raw observations modes: `state_dict` (privileged states) and `sensor_data` (raw sensor data like visual data without postprocessing). `state` is a flat version of `state_dict`. `rgbd` and `pointcloud` apply post-processing on `sensor_data` to give convenient representations of visual data.

The details here show the unbatched shapes. In general there is always a batch dimension unless you are using CPU simulation. Moreover, we annotate what dtype some values are, where some have both a torch and numpy dtype depending on whether you are using GPU or CPU simulation repspectively.

### state_dict

The observation is a dictionary of states. It usually contains privileged information such as object poses. It is not supported for soft-body tasks.

- `agent`: robot proprioception
  - `qpos`: [nq], current joint positions. *nq* is the degree of freedom.
  - `qvel`: [nq], current joint velocities
  <!-- - `base_pose`: [7], robot position (xyz) and quaternion (wxyz) in the world frame -->
  - `controller`: controller states depending on the used controller. Usually an empty dict.
- `extra`: a dictionary of task-specific information, e.g., goal position, end-effector position. This is the return value of a task's `_get_obs_extra` function

### state

It is a flat version of *state_dict*. The observation space is `gym.spaces.Box`.

### sensor_data

In addition to `agent` and `extra`, `sensor_data` and `sensor_param` are introduced.

- `sensor_data`: data captured by sensors configured in the environment
  - `{sensor_uid}`:
    
    If the data comes from a camera sensor:
    - `Color`: [H, W, 4], `torch.uint8`. RGB+Alpha values..
    - `PositionSegmentation`: [H, W, 4], `torch.int16`. The first 3 dimensions stand for (x, y, z) coordinates in the OpenGL/Blender convension. The unit is millimeters. The last dimension represents segmentation ID, see the [Segmentation data section](#segmentation-data) for more details.

- `sensor_param`: parameters of each sensor, which varies depending on type of sensor
  - `{sensor_uid}`:

    If `sensor_uid` corresponds to a camera:
    - `cam2world_gl`: [4, 4], transformation from the camera frame to the world frame (OpenGL/Blender convention)
    - `extrinsic_cv`: [4, 4], camera extrinsic (OpenCV convention)
    - `intrinsic_cv`: [3, 3], camera intrinsic (OpenCV convention)

### rgbd

This observation mode has the same data format as the [sensor_data mode](#sensor_data), but all sensor data from cameras are replaced with the following structure

- `sensor_data`:
  - `{sensor_uid}`:

    If the data comes from a camera sensor:
    - `rgb`: [H, W, 3], `torch.uint8, np.uint8`. RGB.
    - `depth`: [H, W, 1], `torch.int16, np.uint16`. The unit is millimeters. 0 stands for an invalid pixel (beyond the camera far).
    - `segmentation`: [H, W, 1], `torch.int16, np.uint16`. See the [Segmentation data section](#segmentation-data) for more details.

    Otherwise keep the same data without any additional processing as in the sensor_data mode

Note that this data is not scaled/normalized to [0, 1] or [-1, 1] in order to conserve memory, so if you consider to train on RGBD data be sure to scale your data before training on it.

The RGB and depth data visualized can look like below:
```{image} images/replica_cad_rgbd.png
---
alt: RGBD from two cameras of Fetch robot inside the ReplicaCAD dataset scene
---
```

### pointcloud
This observation mode has the same data format as the [sensor_data mode](#sensor_data), but all sensor data from cameras are removed and instead a new key is added called `pointcloud`.

- `pointcloud`:
  - `xyzw`: [N, 4], `torch.float32, np.float32`. Point cloud fused from all cameras in the world frame. "xyzw" is a homogeneous representation. `w=0` for infinite points (beyond the camera far), and `w=1` for the rest.
  - `rgb`: [N, 3], `torch.uint8, np.uint8`. corresponding colors of the fused point cloud
  - `segmentation`: [N, 1], `torch.int16, np.uint16`. See the [Segmentation data section](#segmentation-data) for more details.
Note that the point cloud does not contain more information than RGB-D images unless specified otherwise.

The pointcloud visualized can look like below. Below is the fused pointcloud of two cameras in the ReplicaCAD scene

```{image} images/replica_cad_pcd.png
---
alt: Point cloud of Fetch robot inside the ReplicaCAD dataset scene
---
```

For a quick demo to visualize pointclouds, you can run

```bash
python -m mani_skill.examples.demo_vis_pcd -e "PushCube-v1"
```

### voxel
This observation mode has the same data format as the [sensor_data mode](#sensor_data), but all sensor data from cameras are removed and instead a new key is added called `voxel_grid`.

To use this observation mode, a dictionary of observation config parameters is required to be passed in via obs_mode_config during environment initializations (gym.make()). It should contain the following voxelization config hyperparameters:

- `coord_bounds`: `[torch.float32, torch.float32, torch.float32, torch.float32, torch.float32, torch.float32]` It has form **[x_min, y_min, z_min, x_max, y_max, z_max]**  defining the metric volume to be voxelized.
- `voxel_size`: `torch.int` Defining the side length of each voxel, assuming that all voxels are cubic.
- `device`: `torch.device` The device on which the voxelization takes place.
- `segmentation`: `bool` Defining whether or not to estimate voxel segmentations using the point cloud segmentations. If true then num_channels=11 (including one channel for voxel segmentation), otherwise num_channels=10.

Then, as you step throught the environment you created and get observations, you can see the extra key `voxel_grid` indicating the voxel grid generated:


- `voxel_grid`: `[torch.int, torch.int, torch.int, torch.int, torch.int]` It has form **[N, voxel_size, voxel_size, voxel_size, num_channels]**. Voxel grids generated by fusing the point cloud and rgb data from all cameras. `N` is the batch size. `voxel_size` is the side length of the voxel, as indicated in voxelization configs. `num_channels` indicates the number of feature channels for each voxel. 


The voxel grid can be visualized below. This is an image showing the voxelized scene of PushCube-v1 with slightly-tuned default hyperparameters. The voxel grid is reconstructed from the front camera, following the default camera settings of the PuchCube-v1 task, and hence it only contains the front voxels instead of the voxels throughout the scene.

```{image} images/voxel_pushcube.png
---
alt: Voxelized PushCube-v1 scene at the initial state
---
```

For a quick demo to visualize voxel grids, you can run

<!-- TODO: add command line args -->
```bash
python -m mani_skill.examples.demo_vis_voxel -e "PushCube-v1" --voxel-size 200 --zoom-factor 2.2 --coord-bounds -1 -1 -1 2 2 2
```

Or simply 

```bash
python -m mani_skill.examples.demo_vis_voxel -e "PushCube-v1" 
```

When using just the default settings.

Furthermore, if you use more sensors (currently only RGB and depth cameras) to film the scene and collect more point cloud and RGB data, you can get a more accurate voxel grid reconstruction of the scene.

## Segmentation Data

Objects upon being loaded are automatically assigned a segmentation ID (the `per_scene_id` attribute of `sapien.Entity` objects). To get information about which IDs refer to which Actors / Links, you can run the code below

```python
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.structs import Actor, Link
env = gym.make("PushCube-v1", obs_mode="rgbd")
for obj_id, obj in sorted(env.unwrapped.segmentation_id_map.items()):
    if isinstance(obj, Actor):
        print(f"{obj_id}: Actor, name - {obj.name}")
    elif isinstance(obj, Link):
        print(f"{obj_id}: Link, name - {obj.name}")
```

Note that ID 0 refers to the distant background. For a quick demo of this, you can run

```bash
python -m mani_skill.examples.demo_vis_segmentation -e "PushCube-v1" # plot all segmentations
python -m mani_skill.examples.demo_vis_segmentation -e "PushCube-v1" --id cube # mask everything but the object with name "cube" 
```