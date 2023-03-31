# Environments

[asset-badge]: https://img.shields.io/badge/download%20asset-yes-blue.svg

## Rigid-body

### Pick-and-Place

#### PickCube-v0

(pickcube-v0)=

- Objective: Pick up a cube and move it to a goal position.
- Success metric: The cube is within 2.5 cm of the goal position, and the robot is static.
- Goal specification: 3D goal position.
- Demonstration: 1000 successful trajectories.
- Evaluaion protocol: 100 episodes with different initial joint positions of the robot and initial cube pose.

```{image} thumbnails/PickCube-v0.gif
---
width: 256px
alt: PickCube-v0
---
```

#### StackCube-v0

- Objective: Pick up a red cube and place it onto a green one.
- Success metric: The red cube is placed on top of the green one stably and it is not grasped.
- Demonstration: 1000 successful trajectories.
- Evaluaion protocol: 100 episodes with different initial joint positions of the robot and initial poses of both cubes.

```{image} thumbnails/StackCube-v0.gif
---
width: 256px
alt: StackCube-v0
---
```

#### PickSingleYCB-v0

(picksingleycb-v0)=

![download-asset][asset-badge]

- Objective: Pick up a YCB object and move it to a goal position.
- Success metric: The object is within 2.5 cm of the goal position, and the robot is static.
- Goal specification: 3D goal position.
- Demonstration: 100 successful trajectories for each of the 74 YCB objects.
- Evaluation protocol:
  - 5 episodes per object in the training set (74 YCB objects).
  - 10 episodes per object in the held-out evaluation set (40 objects from other sources).

```{image} thumbnails/PickSingleYCB-v0.gif
---
width: 256px
alt: PickSingleYCB-v0
---
```

Use all models:

```python
env = gym.make("PickSingleYCB-v0")
```

Use a single model:

```python
env = gym.make("PickSingleYCB-v0", model_ids="002_master_chef_can")
```

Use multiple models:

```python
env = gym.make("PickSingleYCB-v0", model_ids=["002_master_chef_can", "003_cracker_box"])
```

Model ids can be found in `mani_skill2/assets/mani_skill2_ycb/info_pick_v0.json`.

#### PickSingleEGAD-v0

![download-asset][asset-badge]

- Objective: Pick up an EGAD object and move it to a goal position.
- Note: The color for the EGAD object is randomized.
- Success metric: The object is within 2.5 cm of the goal position, and the robot is static.
- Goal specification: 3D goal position.
- Demonstration: a total of 7785 trajectories for the 1600 EGAD objects (at most 5 trajectories per object).
- Evaluation protocol:
  - 1 episode per object in the training set (a subset of 150 EGAD objects).
  - 2 episodes per object in the evaluation set (150 held-out EGAD objects).

```{image} thumbnails/PickSingleEGAD-v0.gif
---
width: 256px
alt: PickSingleEGAD-v0
---
```

You can use a similar way as `PickSingleYCB` to select models. Model ids can be found in `mani_skill2/assets/mani_skill2_egad/info_pick_train_v0.json`.

#### PickClutterYCB-v0

![download-asset][asset-badge]

- Objective: Pick up an object from a clutter of 4-8 YCB objects
- Success metric: The object is within 2.5 cm of the goal position, and the robot is static.
- Goal specification: 3D goal position and 3D initial position of the object to pick up (a visible point on the surface).
- Demonstration: a total of 4986 trajectories (5000 training layouts of objects).
- Evaluation protocol:
  - 100 episodes for the training set of YCB objects.
  - 100 episodes for the evaluation set of held-out objects.

```{image} thumbnails/PickClutterYCB-v0.gif
---
width: 256px
alt: PickClutterYCB-v0
---
```

### Assembly

#### PegInsertionSide-v0

- Objective: Insert a peg into the horizontal hole in a box.
- Success metric: Half of the peg is inserted into the hole.
- Demonstration: 1000 successful trajectories.
- Evaluation protocol: 100 episodes with different initial joint positions of the robot, initial poses of the peg and box, the position and size of the hole.

```{image} thumbnails/PegInsertionSide-v0.gif
---
width: 256px
alt: PegInsertionSide-v0
---
```

#### PlugCharger-v0

- Objective: Plug a charger into a wall receptacle.
- Success metric: The charger is fully inserted into the receptacle.
- Demonstration: 1000 successful trajectories.
- Evaluation protocol: 100 episodes with different initial joint positions of the robot, initial poses of the charger and wall.

```{image} thumbnails/PlugCharger-v0.gif
---
width: 256px
alt: PlugCharger-v0
---
```

#### AssemblingKits

![download-asset][asset-badge]

- Objective: Insert an object into the corresponding slot on a board.
- Success metric: An object must fully fit into its slot, which must simultaneously satisfy 3 criteria: (1) height of the object center is within 3mm of the height of the board; (2) rotation error is within 4 degrees; (3) position error is within 2cm.
- Demonstration: a total of 1720 trajectories for 337 kit configurations and 20 objects.
- Evaluation protocol:
  - 100 episodes for 20 training objects.
  - 100 episodes for 20 held-out evaluation objects.

```{image} thumbnails/AssemblingKits-v0.gif
---
width: 256px
alt: AssemblingKits-v0
---
```

### Miscellaneous

#### PandaAvoidObstacles-v0

![download-asset][asset-badge]

- Objective: Navigate the (Panda) robot arm through a region of dense obstacles and move the end-effector to a goal pose.
- Note: The shape and color of dense obstacles are randomized.
- Success metric: The end-effector pose is within 2.5 cm and 15 degrees of the goal pose.
- Goal specification: The goal pose of the end-effector.
- Demonstration: 1976 successful trajectories.
- Evaluation protocol: 100 episodes with different layouts of obstacles.

```{image} thumbnails/PandaAvoidObstacles-v0.gif
---
width: 256px
alt: PandaAvoidObstacles-v0
---
```

#### TurnFaucet-v0

![download-asset][asset-badge]

- Objective: Turn on a faucet by rotating its handle.
- Success metric: The faucet handle has been turned past a target angular distance.
- Goal specification: The remaining angular distance to rotate the handle, the initial center of mass of the target handle (since there can be multiple handles in a single faucet), and the direction to rotate the handle specified as 3D joint axis.
- Demonstration: a total of 5510 trajectories (100 trajectories per faucet for most of 60 models from PartNet-Mobility).
- Evaluation protocol:
  - 5 episodes per model in the training set (60)
  - 17 episodes per model in the test set (18)

```{image} thumbnails/TurnFaucet-v0.gif
---
width: 256px
alt: TurnFaucet-v0
---
```

Use all models:

```python
env = gym.make("TurnFaucet-v0")
```

Use a single model:

```python
env = gym.make("TurnFaucet-v0", model_ids="5000")
```

Use multiple models:

```python
env = gym.make("TurnFaucet-v0", model_ids=["5001", "5002"])
```

Model ids can be found in `mani_skill2/assets/partnet_mobility/meta/info_faucet_train.json`.

### Mobile Manipulation from ManiSkill1

#### OpenCabinetDoor-v1

(opencabinetdoor-v1)=

![download-asset][asset-badge]

- Objective: A single-arm mobile robot needs to open a designated target door on a cabinet.
- Note: The friction and damping parameters for the door joints are randomized.
- Success metric: The target door has been opened to at least 90\% of the maximum range, and the target door is static.
- Goal specification: The target angular distance to rotate the door, the initial center of mass of the target link (since there can be multiple doors in a single cabinet), and the direction to rotate the door specified as 3D joint axis.
- Demonstration: 300 trajectories for each target door in the training object set. The training object set consists of 42 cabinets. Each cabinet could contain multiple doors.
- Evaluation protocol:
  - 125 episodes for models in the training set (42)
  - 25 episodes per model in the test set (10)

```{image} thumbnails/OpenCabinetDoor-v1.gif
---
width: 256px
alt: OpenCabinetDoor-v1
---
```

#### OpenCabinetDrawer-v1

![download-asset][asset-badge]

- Objective: A single-arm mobile robot needs to open a designated target drawer on a cabinet.
- Note: The friction and damping parameters for the drawer joints are randomized.
- Success metric: The target drawer has been opened to at least 90\% of the maximum range, and the target drawer is static.
- Goal specification: The target linear distance to pull the drawer, the initial center of mass of the target link (since there can be multiple drawers in a single cabinet), and the direction to pull the drawer specified as 3D joint axis.
- Demonstration: 300 trajectories for each target drawer in the training object set. The training object set consists of 25 cabinets. Each cabinet could contain multiple drawers.
- Evaluation protocol:
  - 5 episodes per model in the training set (25)
  - 25 episodes per model in the test set (10)

```{image} thumbnails/OpenCabinetDrawer-v1.gif
---
width: 256px
alt: OpenCabinetDrawer-v1
---
```

#### PushChair-v1

(pushchair-v1)=

![download-asset][asset-badge]

- Objective: A dual-arm mobile robot needs to push a swivel chair to a target location on the ground (indicated by a red hemisphere) and prevent it from falling over.
- Note: The friction and damping parameters for the chair joints are randomized.
- Success metric: The chair is close enough (within 15 cm) to the target location, is static, and does not fall over.
- Demonstration: 300 trajectories for each chair in the training object set. The training object set consists of 26 chairs.
- Evaluation protocol:
  - 5 episodes per model in the training set (26)
  - 25 episodes per model in the test set (10)

```{image} thumbnails/PushChair-v1.gif
---
width: 256px
alt: PushChair-v1
---
```

#### MoveBucket-v1

![download-asset][asset-badge]

- Objective: A dual-arm mobile robot needs to move a bucket with a ball inside and lift it onto a platform.
- Success metric: The bucket is placed on or above the platform at the upright position, and the bucket is static, and the ball remains in the bucket.
- Demonstration: 300 trajectories for each bucket in the training object set. The training object set consists of 29 buckets.
- Evaluation protocol:
  - 5 episodes per model in the training set (29)
  - 25 episodes per model in the test set (10)

```{image} thumbnails/MoveBucket-v1.gif
---
width: 256px
alt: MoveBucket-v1
---
```

## Soft-body

### Excavate-v0

- Objective: Lift a specific amount of clay to a target height.
- Success metric: The amount of lifted clay must be within a given range; the lifted clay is higher than a specific height; fewer than 20 clay particles are spilled on the ground; soft body velocity <0.05.
- Goal specification: Target clay amount.
- Demonstration: 200 successful trajectories.
- Evaluation protocol: 100 episodes with different bucket poses and initial heightmaps of clay.

```{image} thumbnails/Excavate-v0.gif
---
width: 256px
alt: Excavate-v0
---
```

### Fill-v0

- Objective: Fill clay from a bucket into the target beaker.
- Success metric: The amount of clay inside the target beaker >90\%; soft body velocity <0.05.
- Goal specification: Beaker position.
- Demonstration: 200 successful trajectories.
- Evaluation protocol: 100 episodes with different initial rotations of the bucket and initial positions of the beaker.

```{image} thumbnails/Fill-v0.gif
---
width: 256px
alt: Fill-v0
---
```

### Pour-v0

- Objective: Pour liquid from a bottle into a beaker.
- Success metric: The liquid level in the beaker is within 4mm of the red line; the spilled water is fewer than 100 particles; the bottle returns to the upright position in the end; robot arm velocity <0.05.
- Goal specification: Red line height.
- Demonstration: 200 successful trajectories.
- Evaluation protocol: 100 episodes with different bottle positions, the level of water in the bottle, and beaker positions.

```{image} thumbnails/Pour-v0.gif
---
width: 256px
alt: Pour-v0
---
```

### Hang-v0

- Objective: Hang a noodle on a target rod.
- Success metric: Part of the noodle is higher than the rod; two ends of the noodle are on different sides of the rod; the noodle is not touching the ground; the gripper is open; soft body velocity <0.05.
- Goal specification: Rod position.
- Demonstration: 200 successful trajectories.
- Evaluation protocol: 100 episodes with different initial positions of the gripper and rod poses.

```{image} thumbnails/Hang-v0.gif
---
width: 256px
alt: Hang-v0
---
```

### Pinch-v0

![download-asset][asset-badge]

- Objective: Deform plasticine into a target shape.
- Success metric: The Chamfer distance between the current plasticine and the target shape is less than $0.3t$, where $t$ is the Chamfer distance between the initial shape and target shape.
- Goal specification: RGBD / point cloud observations of the target plasticine from 4 different views.
- Demonstration: 1556 successful trajectories. Different trajectories correspond to different target shapes.
- Evaluation protocol: 50 episodes with different target shapes.

```{image} thumbnails/Pinch-v0.gif
---
width: 256px
alt: Pinch-v0
---
```

### Write-v0

![download-asset][asset-badge]

- Objective: Write a given character on clay. The target character is randomly sampled from an alphabet of over 50 characters.
- Success metric: The IoU (Intersection over Union) between the current pattern and the target character is larger than 0.8.
- Goal specification: The depth map of the target character.
- Demonstration: 200 successful trajectories.
- Evaluation protocol: 50 episodes with different target characters.

```{image} thumbnails/Write-v0.gif
---
width: 256px
alt: Write-v0
---
```
