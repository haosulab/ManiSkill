# Tasks

ManiSkill features a number of built-in rigid-body tasks, all GPU parallelized and demonstrate a range of features.

Soft-body tasks will be added back in as they are still in development as part of a new soft-body simulator we are working on.


For each task documented here we provide a "Task Card" which briefly describes all the important aspects of the task, including task description, supported robots, randomizations, success/fail conditions, and goal specification in observations. We further show tags describing whether there are dense rewards provided, and if assets need to be downloaded via 

```python -m mani_skill.utils.download_asset <env_id>```

Note that some tasks do not have goal specifications. This generally means part of the observation (e.g. cube pose or the visual observation) indicates the goal for you.

This is still a WIP as we add in more tasks and document more things.

[asset-badge]: https://img.shields.io/badge/download%20asset-yes-blue.svg
[reward-badge]: https://img.shields.io/badge/dense%20reward-yes-green.svg

## Table-Top Rigid-body Tasks

### PickCube-v1
![dense-reward][reward-badge]

:::{dropdown} Task Card
:icon: note
:color: primary

**Task Description:**
A simple task where the objective is to grasp a red cube and move it to a target goal position.

**Supported Robots: Panda, Fetch**

**Randomizations:**
- the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
- the cube's z-axis rotation is randomized to a random angle
- the target goal position (marked by a green sphere) of the cube has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

**Success Conditions:**
- the cube position is within `goal_thresh` (default 0.025m) euclidean distance of the goal position
- the robot is static (q velocity < 0.2)

**Goal Specification:**
- 3D goal position (also visualized in human renders)
:::

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCube-v1_rt.mp4" type="video/mp4">
</video>

### StackCube-v1
![dense-reward][reward-badge]
:::{dropdown} Task Card
:icon: note
:color: primary

**Task Description:**
The goal is to pick up a red cube and stack it on top of a green cube and let go of the cube without it falling

**Supported Robots: Panda, Fetch, xArm**

**Randomizations:**
- both cubes have their z-axis rotation randomized
- both cubes have their xy positions on top of the table scene randomized. The positions are sampled such that the cubes do not collide with each other

**Success Conditions:**
- the red cube is on top of the green cube (to within half of the cube size)
- the red cube is static
- the red cube is not being grasped by the robot (robot must let go of the cube)
:::

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/StackCube-v1_rt.mp4" type="video/mp4">
</video>

### PickSingleYCB-v1
![download-asset][asset-badge] ![dense-reward][reward-badge]

:::{dropdown} Task Card
:icon: note
:color: primary

**Task Description:**
Pick up a random object sampled from the [YCB dataset](https://www.ycbbenchmarks.com/) and move it to a random goal position

**Supported Robots: Panda, Fetch, xArm**

**Randomizations:**
- the object's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
- the object's z-axis rotation is randomized
- the object geometry is randomized by randomly sampling any YCB object. (during reconfiguration)

**Success Conditions:**
- the object position is within goal_thresh (default 0.025) euclidean distance of the goal position
- the robot is static (q velocity < 0.2)

**Goal Specification:**
- 3D goal position (also visualized in human renders)

**Additional Notes**
- On GPU simulation, in order to collect data from every possible object in the YCB database we recommend using at least 128 parallel environments or more, otherwise you will need to reconfigure in order to sample new objects.
:::


<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickSingleYCB-v1_rt.mp4" type="video/mp4">
</video>

### PegInsertionSide-v1

:::{dropdown} Task Card
:icon: note
:color: primary

**Task Description:**
Pick up a orange-white peg and insert the orange end into the box with a hole in it.

**Supported Robots: Panda, Fetch, xArm**

**Randomizations:**
- Peg half length is randomized between 0.085 and 0.125 meters. Box half length is the same value. (during reconfiguration)
- Peg radius/half-width is randomized between 0.015 and 0.025 meters. Box hole's radius is same value + 0.003m of clearance. (during reconfiguration)
- Peg is laid flat on table and has it's xy position and z-axis rotation randomized
- Box is laid flat on table and has it's xy position and z-axis rotation randomized

**Success Conditions:**
- The white end of the peg is within 0.015m of the center of the box (inserted mid way).

:::

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PegInsertionSide-v1_rt.mp4" type="video/mp4">
</video>

### LiftPegUpright-v1
:::{dropdown} Task Card
:icon: note
:color: primary

**Task Description:**
A simple task where the objective is to move a peg laying on the table to any upright position on the table

**Supported Robots: Panda, Fetch, xArm**

**Randomizations:**
- the peg's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat along it's length on the table

**Success Conditions:**
- the absolute value of the peg's z euler angle is within 0.08 of $\pi$/2 and the z position of the peg is within 0.005 of its half-length (0.12).
:::

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/LiftPegUpright-v1_rt.mp4" type="video/mp4">
</video>

### PushCube-v1
![dense-reward][reward-badge]
:::{dropdown} Task Card
:icon: note
:color: primary

**Task Description:**
A simple task where the objective is to push and move a cube to a goal region in front of it

**Supported Robots: Panda, Fetch, xArm**

**Randomizations:**
- the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
- the target goal region is marked by a red/white circular target. The position of the target is fixed to be the cube xy position + [0.1 + goal_radius, 0]

**Success Conditions:**
- the cube's xy position is within goal_radius (default 0.1) of the target's xy position by euclidean distance.
:::

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PushCube-v1_rt.mp4" type="video/mp4">
</video>

### PullCube-v1
:::{dropdown} Task Card
:icon: note
:color: primary

**Task Description:**
A simple task where the objective is to pull a cube onto a target.

**Supported Robots: Panda, Fetch, xArm**

**Randomizations:**
- the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1].
- the target goal region is marked by a red and white target. The position of the target is fixed to be the cube's xy position - [0.1 + goal_radius, 0]

**Success Conditions:**
- the cube's xy position is within goal_radius (default 0.1) of the target's xy position by euclidean distance.
:::

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PullCube-v1_rt.mp4" type="video/mp4">
</video>

### AssemblingKits-v1
![download-asset][asset-badge]

:::{dropdown} Task Card
:icon: note
:color: primary

**Task Description:**
The robot must pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.

**Supported Robots: Panda with RealSense wrist camera**

**Randomizations:**
- the kit geometry is randomized, with different already inserted shapes and different holes affording insertion of specific shapes. (during reconfiguration)
- the misplaced shape's geometry is sampled from one of 20 different shapes. (during reconfiguration)
- the misplaced shape is randomly spawned anywhere on top of the board with a random z-axis rotation

**Success Conditions:**
- the cube's xy position is within goal_radius (default 0.1) of the target's xy position by euclidean distance.
:::

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/AssemblingKits-v1_rt.mp4" type="video/mp4">
</video>

## Table-Top Dexterous Hand Tasks

### RotateValveLevel0-v1

(Note there is Level0, Level1, ... to Level4)

:::{dropdown} Task Card
:icon: note
:color: primary

**Task Description:**
Using the D'Claw robot, rotate a [ROBEL valve](https://sites.google.com/view/roboticsbenchmarks/platforms/dclaw)

**Supported Robots: DClaw**

**Randomizations:**
- Rotation direction $r$. Level 0: $r=1$, Level 4: $r \in \{1, -1\}$. (during reconfiguration)
- Number of valves on the ROBEL valve $v$. Level 0-1: $v=3$, Level 2-4: $v \in [3, 6]$. (during reconfiguration)
- Valve angles $\phi$. Level 0: Equally spaced $\phi = (0, 2\pi/3, 4\pi/3)$, Level 1-4: Each angle is randomized. (during reconfiguration)
- Level 4 only: valve radii are randomized a little. (during reconfiguration)
- Initial valve rotation is randomized

**Success Conditions:**
- The valve rotated more than $\theta$ radians from its initial position. Level 0: $\theta = \pi/2$, Level 1-3: $\theta = \pi$, Level 4: $\theta=2\pi$

**Goal Specification:**
- Rotation direction $r$ which can be 1 or -1. Note that the amount to rotate is implicit and depends of level

:::{figure} https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/RotateValve-v1_rt.png
:alt: rotate valve task
:::


### RotateCubeLevel0-v1

(Note there is Level0, Level1, ... to Level4)

:::{dropdown} Task Card
:icon: note
:color: primary

**Task Description:**
Using the TriFingerPro robot, rotate a cube

**Supported Robots: TriFingerPro**

**Randomizations:**
- Level 0: Random goal position on the table, no orientation.
- Level 1:  Random goal position on the table, including yaw orientation.
- Level 2: Fixed goal position in the air with x,y = 0.  No orientation.
- Level 3: Random goal position in the air, no orientation.
- Level 4: Random goal pose in the air, including orientation.

**Success Conditions:**
- The rotated cube should be within 0.02 m of the goal position and 0.1 rad of the goal orientation.

## Control Tasks

### MS-CartPole-v1


:::{dropdown} Task Card
:icon: note
:color: primary

**Task Description:**
Keep the CartPole stable and up right by sliding it left and right

**Supported Robots: None**

**Randomizations:**
- TODO

**Success Conditions:**
- the cart is within 0.25m of the center of the rail (which is at 0)
- the cosine of the hinge angle attaching the pole is between 0.995 and 1

**Goal Specification:**
- None

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/MS-CartPole-v1_rt.mp4" type="video/mp4">
</video>