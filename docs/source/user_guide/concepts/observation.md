# Observation

<!-- See our [colab tutorial](https://colab.research.google.com/github/haosulab/ManiSkill2/blob/main/examples/tutorials/customize_environments.ipynb#scrollTo=NaSQ7CD2sswC) for how to customize cameras. -->

## Observation mode

**The observation mode defines the observation space.**
All ManiSkill2 tasks take the observation mode (`obs_mode`) as one of the input arguments of `__init__`.
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
    - `PositionSegmentation`: [H, W, 4], `torch.int16`. The first 3 dimensions stand for (x, y, z) coordinates in the OpenGL/Blender convension. The unit is millimeters. The last dimension represents segmentation ID. 

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
  - `rgb`: [N, 3], `torch.uint8, np.uint8` corresponding colors of the fused point cloud

Note that the point cloud does not contain more information than RGB-D images unless specified otherwise.

The pointcloud visualized can look like below. Below is the fused pointcloud of two cameras in the ReplicaCAD scene

```{image} images/replica_cad_pcd.png
---
alt: Point cloud of Fetch robot inside the ReplicaCAD dataset scene
---
```
<!-- 
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

### More Details on Mesh and Actor-Level segmentations

An "actor" is a fundamental object that represents a physical entity (rigid body) that can be simulated in SAPIEN (the backend of ManiSkill). An articulated object is a collection of links interconnected by joints, and each link is also an actor. In SAPIEN, `scene.get_all_actors()` will return all the actors that are not links of articulated objects. The examples are the ground, the cube in [PickCube](./environments.md#PickCube-v1), and the YCB objects in [PickSingleYCB](./environments.md#picksingleycb-v0). `scene.get_all_articulations()` will return all the articulations. The examples are the robots, the cabinets in [OpenCabinetDoor](./environments.md#opencabinetdoor-v1), and the chairs in [PushChair](./environments.md#pushchair-v1). Below is an example of how to get actors and articulations in SAPIEN.

```python
import sapien

scene: sapien.Scene = ...
actors = scene.get_all_actors()  # actors excluding links
articulations = scene.get_all_articulations()  # articulations
for articulation in articulations:
  links = articulation.get_links()  # links of an articulation
```

In ManiSkill, our environments provide interfaces to wrap the above SAPIEN functions:

- `env.get_actors()`: return all task-relevant actors excluding links. Note that some actors might be excluded from `env._scene.get_all_actors()`.
- `env.get_articulations()`: return all task-relevant articulations. Note that some articulations might be excluded from `env._scene.get_all_articulations()`.

```{eval-rst}
.. subfigure:: AB
  :subcaptions: below
  :class-grid: outline
  :align: center

  .. image:: https://sapien.ucsd.edu/docs/latest/_images/label1.png
    :alt: Actor-level segmentation
    :width: 256px

  .. image:: https://sapien.ucsd.edu/docs/latest/_images/label0.png
    :alt: Mesh-level segmentation
    :width: 256px
```

The segmentation image is a `[H, W, 4]` array. The second channel corresponds to the ids of actors. The first channel corresponds to the ids of visual meshes (each actor can consist of multiple visual meshes).
Thus, given the actors, you can use the ids of these actors (`actor.id`) to query the actor segmentation to segment out a particular object. For example,

```python
import mani_skill.envs, gymnasium as gym
import numpy as np

env = gym.make('PickSingleYCB-v0', obs_mode='rgbd', camera_cfgs={'add_segmentation': True})
obs, _ = env.reset()

print(env.get_actors()) # e.g., [Actor(name="ground", id="14"), Actor(name="008_pudding_box", id="15"), Actor(name="goal_site", id="16")]
print([x.name for x in env.get_articulations()]) # ['panda_v2']

# get the actor ids of objects to manipulate; note that objects here are not articulated
target_object_actor_ids = [x.id for x in env.get_actors() if x.name not in ['ground', 'goal_site']]

# get the robot link ids (links are subclass of actors)
robot_links = env.agent.robot.get_links() # e.g., [Actor(name="root", id="1"), Actor(name="root_arm_1_link_1", id="2"), Actor(name="root_arm_1_link_2", id="3"), ...]
robot_link_ids = np.array([x.id for x in robot_links], dtype=np.int32)

# obtain segmentations of the target object(s) and the robot
for camera_name in obs['image'].keys():
    seg = obs['image'][camera_name]['Segmentation'] # (H, W, 4); [..., 0] is mesh-level; [..., 1] is actor-level; [..., 2:] is zero (unused)
    actor_seg = seg[..., 1]
    new_seg = np.zeros_like(actor_seg)
    new_seg[np.isin(actor_seg, robot_link_ids)] = 1
    for i, target_object_actor_id in enumerate(target_object_actor_ids):
        new_seg[np.isin(actor_seg, target_object_actor_id)] = 2 + i
    obs['image'][camera_name]['new_seg'] = new_seg
    # print(np.unique(new_seg))
```

However, the actor segmentations do not contain finegrained information on object parts, such as handles and door surfaces of cabinets. In this case, you need to leverage the mesh-level segmentations and the fine-grained visual bodies of actors to obtain the segmentations to e.g., handles and door surfaces. For example,

```python
import mani_skill.envs, gymnasium as gym
import numpy as np

env = gym.make('OpenCabinetDoor-v1', obs_mode='rgbd', camera_cfgs={'add_segmentation': True})
obs, _ = env.reset()

print(env.get_actors()) # e.g., [Actor(name="ground", id="20"), Actor(name="visual_ground", id="21")], which are not very helpful
print([x.name for x in env.get_articulations()]) # e.g., ['mobile_panda_single_arm', '1017']

# We'd like to obtain fine-grained part segmentations such as handles, so we need to obtain the finegrained visual bodies that correspond to each cabinet link

# get the names and ids of cabinet visual bodies and manually group them based on semantics
cabinet_links = env.cabinet.get_links() # e.g., [Actor(name="base", id="22"), Actor(name="link_1", id="23"), Actor(name="link_0", id="24")]
cabinet_visual_bodies = [x.get_visual_bodies() for x in cabinet_links]
cabinet_visual_body_names = np.concatenate([[b.name for b in cvbs] for cvbs in cabinet_visual_bodies]) # e.g., array(['shelf-10', 'shelf-11', 'frame_horizontal_bar-26', ...])
cabinet_visual_body_ids = np.concatenate([[b.get_visual_id() for b in cvbs] for cvbs in cabinet_visual_bodies]).astype(np.int32) # e.g., array([15, 16, 17, 18, 19, ...])
        
all_handle_ids = []
all_door_ids = []
all_drawer_ids = []
all_cabinet_rest_ids = []
for i in range(len(cabinet_visual_body_names)):
    # print(cabinet_visual_body_names[i], cabinet_visual_body_ids[i])
    if 'handle' in cabinet_visual_body_names[i]:
        all_handle_ids.append(cabinet_visual_body_ids[i])
    elif 'door_surface' in cabinet_visual_body_names[i]:
        all_door_ids.append(cabinet_visual_body_ids[i])
    elif 'drawer_front' in cabinet_visual_body_names[i]:
        all_drawer_ids.append(cabinet_visual_body_ids[i])
    else:
        all_cabinet_rest_ids.append(cabinet_visual_body_ids[i])
all_handle_ids = np.array(all_handle_ids)
all_door_ids = np.array(all_door_ids)
all_drawer_ids = np.array(all_drawer_ids)
all_cabinet_rest_ids = np.array(all_cabinet_rest_ids)
                
# get the robot link ids
robot_links = env.agent.robot.get_links() # e.g., [Actor(name="root", id="1"), Actor(name="root_arm_1_link_1", id="2"), Actor(name="root_arm_1_link_2", id="3"), ...]
robot_link_ids = np.array([x.id for x in robot_links], dtype=np.int32)

# get the segmentations of different cabinet parts and the robots
for camera_name in obs['image'].keys():
    seg = obs['image'][camera_name]['Segmentation'] # (H, W, 4); [..., 0] is mesh-level; [..., 1] is actor-level; [..., 2:] is zero (unused)
    mesh_seg = seg[..., 0]
    actor_seg = seg[..., 1]
    
    # visual body ids correspond to mesh-level segmentations
    semantic_grouped_seg = np.zeros_like(mesh_seg)
    semantic_grouped_seg[np.isin(mesh_seg, all_handle_ids)] = 1
    semantic_grouped_seg[np.isin(mesh_seg, all_door_ids)] = 2
    semantic_grouped_seg[np.isin(mesh_seg, all_drawer_ids)] = 3
    semantic_grouped_seg[np.isin(mesh_seg, all_cabinet_rest_ids)] = 4
    
    # link ids correspond to actor-level segmentations, since a link is a subclass of an actor
    semantic_grouped_seg[np.isin(actor_seg, robot_link_ids)] = 5
    
    obs['image'][camera_name]['semantic_grouped_seg'] = semantic_grouped_seg
    print(f"Summary of # points in the processed segmentaion map for camera {camera_name}:", [(semantic_grouped_seg == x).sum() for x in range(6)])
``` -->
