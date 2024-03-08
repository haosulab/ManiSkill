# Tasks

ManiSkill features a number of built-in rigid-body tasks, all GPU parallelized and demonstrate a range of features.

Soft-body tasks will be added back in as they are still in development as part of a new soft-body simulator we are working on.


For each task documented here we provide a "Task Card" which briefly describes all the important aspects of the task, including task description, supported robots, randomizations, success/fail conditions, and goal specification in observations.

Note that some tasks do not have goal specifications. This generally means part of the observation (e.g. cube pose) indicates the goal for you.

This is still a WIP as we add in more tasks and document more things.

[asset-badge]: https://img.shields.io/badge/download%20asset-yes-blue.svg

## Rigid-body

#### PickCube-v1


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


<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill2/raw/dev/figures/environment_demos/pick_cube_rt.mp4" type="video/mp4">
</video>

#### StackCube-v1

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

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill2/raw/dev/figures/environment_demos/stack_cube_rt.mp4" type="video/mp4">
</video>


#### PushCube-v1

**Task Description:**
A simple task where the objective is to push and move a cube to a goal region in front of it

**Supported Robots: Panda, Fetch, xArm**

**Randomizations:**
- the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
- the target goal region is marked by a red/white circular target. The position of the target is fixed to be the cube xy position + [0.1 + goal_radius, 0]

**Success Conditions:**
- the cube's xy position is within goal_radius (default 0.1) of the target's xy position by euclidean distance.


<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill2/raw/dev/figures/environment_demos/push_cube_rt.mp4" type="video/mp4">
</video>

#### PickSingleYCB-v1
![download-asset][asset-badge]

**Task Description:**
Pick up a random object sampled from the [YCB dataset](https://www.ycbbenchmarks.com/) and move it to a random goal position

**Supported Robots: Panda, Fetch, xArm**

**Randomizations:**
- the object's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
- the object's z-axis rotation is randomized
- the object geometry is randomized by randomly sampling any YCB object

**Success Conditions:**
- the object position is within goal_thresh (default 0.025) euclidean distance of the goal position
- the robot is static (q velocity < 0.2)

**Goal Specification:**
- 3D goal position (also visualized in human renders)

**Additional Notes**
- On GPU simulation, in order to collect data from every possible object in the YCB database we recommend using at least 128 parallel environments or more, otherwise you will need to reconfigure in order to sample new objects.


<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill2/raw/dev/figures/environment_demos/pick_single_ycb_rt.mp4" type="video/mp4">
</video>