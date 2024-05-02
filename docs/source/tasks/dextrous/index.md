# Dextrous Hand Tasks
[asset-badge]: https://img.shields.io/badge/download%20asset-yes-blue.svg
[reward-badge]: https://img.shields.io/badge/dense%20reward-yes-green.svg
## Table-Top Dexterous Hand Tasks

### RotateValveLevel0-v1
![dense-reward][reward-badge]

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
:::

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/RotateValveLevel1-v1_rt.mp4" type="video/mp4">
</video>



### RotateSingleObjectInHandLevel0-v1
![dense-reward][reward-badge]

(Note there is Level0, Level1, ... to Level4)

:::{dropdown} Task Card
:icon: note
:color: primary

**Task Description:**
Using the allegro hand with tactile sensors, rotate a random object.

**Supported Robots: Allegro Hand**

**Randomizations:**
WIP

**Success Conditions:**
WIP

**Goal Specification:**
WIP
:::

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/RotateSingleObjectInHandLevel3-v1_rt.mp4" type="video/mp4">
</video>

### TriFingerRotateCubeLevel0-v1
![dense-reward][reward-badge]

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
:::