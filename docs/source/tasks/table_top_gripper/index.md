# Table-Top 2 Finger Gripper Tasks

[asset-badge]: https://img.shields.io/badge/download%20asset-yes-blue.svg
[reward-badge]: https://img.shields.io/badge/dense%20reward-yes-green.svg

These are tasks situated on table and involve a two-finger gripper arm robot manipulating objects on the surface.

## PickCube-v1
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

## PlugCharger-v1

:::{dropdown} Task Card
:icon: note
:color: primary

**Task Description:**
The robot must pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.

**Supported Robots: Panda with RealSense wrist camera**

**Randomizations:**
- The charger position is randomized on the XY plane on top of the table. The rotation is also randomized
- The receptacle position is randomized on the XY plane and the rotation is also randomized. Note that the human render camera has its pose
fixed relative to the receptacle.

**Success Conditions:**
- The charger is inserted into the receptacle
:::

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PlugCharger-v1_rt.mp4" type="video/mp4">
</video>

## StackCube-v1
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

## PickSingleYCB-v1
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

## PegInsertionSide-v1
![dense-reward][reward-badge]

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

## LiftPegUpright-v1
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

## PushCube-v1
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

## PullCube-v1
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

## AssemblingKits-v1
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


## TwoRobotPickCube-v1
![dense-reward][reward-badge]


:::{dropdown} Task Card
:icon: note
:color: primary

**Task Description:**
One robot must push/give the cube on its side to the other side so the other robot can pick it up and bring it to a goal

**Supported Robots: (Panda, Panda)**

**Randomizations:**
- cube position is spawned on one side and randomized on the XY plane.

**Success Conditions:**
- the cube's xy position is within 0.025 meteres of the goal position by euclidean distance.
:::

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/TwoRobotPickCube-v1_rt.mp4" type="video/mp4">
</video>


## TwoRobotStackCube-v1
![dense-reward][reward-badge]

:::{dropdown} Task Card
:icon: note
:color: primary

**Task Description:**
One robot must push/give the cube on its side to the other side so the other robot can stack another cube on top of it.

**Supported Robots: (Panda, Panda)**

**Randomizations:**
- two cubes are spawned on both sides of the table and have their positon randomized on the XY plane

**Success Conditions:**
- the blue cube is on top of the green cube and neither cube is being grasped by any robot
:::

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/TwoRobotStackCube-v1_rt.mp4" type="video/mp4">
</video>


## PokeCube-v1
![dense-reward][reward-badge]

:::{dropdown} Task Card
:icon: note
:color: primary

**Task Description:**
A simple task where the objective is to poke a red cube with a peg and push it to a target goal position.

**Supported Robots: Panda, Fetch, xArm**

**Randomizations:**
- the peg's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat along it's length on the table
- the cube's x-coordinate is fixed to peg's x-coordinate + peg half-length (0.12) + 0.1 and y-coordinate is randomized in range [-0.1, 0.1]. It is placed flat on the table
- the cube's z-axis rotation is randomized in range [-$\pi$/ 6, $\pi$ / 6]
- the target goal region is marked by a red/white circular target. The position of the target is fixed to be the cube xy position + [0.05 + goal_radius, 0]

**Success Conditions:**
- the cube's xy position is within goal_radius (default 0.05) of the target's xy position by euclidean distance
- the face of peg's head is close to the cube's face and the orientation is aligned
- the peg's z position is within 0.005 of its half-width (0.25), that is close the table

:::

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PokeCube-v1_rt.mp4" type="video/mp4">
</video>

## PlaceSphere-v1
![dense-reward][reward-badge]

:::{dropdown} Task Card
:icon: note
:color: primary

**Task Description:**
A simple task where the objective is to grasp a sphere and place it on top of a little bin.

**Supported Robots: Panda**

**Randomizations:**
- the sphere's xy position is randomized on top of a table in the region [-0.1, -0.05] x [-0.1, 0.1]. It is placed at the first 1/4 zone along the x-axis
- the bin's xy position is randomized on top of a table in the region [0.0, 1.0] x [-0.1, 0.1]. It is placed at the last 1/2 zone along the x-axis so that it doesn't collide the sphere

**Success Conditions:**
- the sphere is on top of the bin. That is, the sphere's xy-distance to the bin goes near 0, and its z-distance to the bin goes near the sphere radius + the bottom bin block's side length
- the object is static. That is, its linear and angular velocities are bounded with a small value
- the gripper is not grasping the object
:::

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PlaceSphere-v1_rt.mp4" type="video/mp4">
</video>
  
## RollBall-v1
![dense-reward][reward-badge]

:::{dropdown} Task Card
:icon: note
:color: primary

**Task Description:**
A simple task where the objective is to push and roll a ball to a goal region at the other end of the table.

**Supported Robots: (Panda)**

**Randomizations:**
- the ball's xy position is randomized on top of a table in the region [0.2, 0.5] x [-0.4, 0.7]. It is placed flat on the table
- the target goal region is marked by a red/white circular target. The position of the target is randomized on top of a table in the region [-0.4, -0.7] x [0.2, -0.9]

**Success Conditions:**
-  the ball's xy position is within goal_radius (default 0.1) of the target's xy position by euclidean distance.
:::

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/RollBall-v1_rt.mp4" type="video/mp4">
</video>

## PushT-v1
![dense-reward][reward-badge]

:::{dropdown} Task Card
:icon: note
:color: primary

**Task Description:**
Digital Twin of real life push-T task from Diffusion Policy: https://diffusion-policy.cs.columbia.edu/

"In this task, the robot needs to \
1 precisely push the T- shaped block into the target region, and \
2 move the end-effector to the end-zone which terminates the episode." \
[We do not require 1 for the digital task]

**Supported Robots: PandaStick (WIP UR5e)**

**Randomizations:**
- The 3D T block's initial center of mass is randomized in the region on the table: [-1,1] x [-1,2] + T Goal initial position. It is placed flat on the table
- The 3D T block's initial z rotation is randomized in [0,2pi] around the center of mass of the block

**Success Conditions:**
- The 3D T block covers at least 90% of the 2D T goal zone
:::

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PushT-v1_rt.mp4" type="video/mp4">
</video>