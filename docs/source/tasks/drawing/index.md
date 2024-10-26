# Drawing Tasks

ManiSkill has example environments that simulate "drawing" with a robot. Currently there are no reward functions / success conditions, just base environments to play around with and extend for your own use cases.

## TableTopFreeDraw-v1

:::{dropdown} Task Card
:icon: note
:color: primary

**Task Description:**
Instantiates a table with a white canvas on it and a robot with a stick that draws red lines.

**Supported Robots: PandaStick**

**Randomizations:**
None

**Success Conditions:**
None

**Goal Specification:**
None
:::

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/TableTopFreeDraw-v1_rt.mp4" type="video/mp4">
</video>

## DrawTriangle-v1

:::{dropdown} Task Card
:icon: note
:color: primary

**Task Description:**
Instantiates a table with a white canvas on it and a goal triangle with an outline. A robot with a stick is to draw the triangle with a red line.

**Supported Robots: PandaStick**

**Randomizations:**
- the goal triangle's position on the xy-plane is randomized
- the goal triangle's z-rotation is randomized in range [0, 2 $\pi$]

**Success Conditions:**
- the drawn points by the robot are within a euclidean distance of 0.05m with points on the goal triangle
:::

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/figures/environment_demos/DrawTriangle-v1_rt.mp4" type="video/mp4">
</video>