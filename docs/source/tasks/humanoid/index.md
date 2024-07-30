# Humanoid Tasks

[asset-badge]: https://img.shields.io/badge/download%20asset-yes-blue.svg
[reward-badge]: https://img.shields.io/badge/dense%20reward-yes-green.svg

Both real-world humanoids and the Mujoco humanoid are supported in ManiSkill, and we are still in the process of adding more tasks


## UnitreeG1PlaceAppleInBowl-v1
![dense-reward][reward-badge]

:::{dropdown} Task Card
:icon: note
:color: primary

**Task Description:**
Control the humanoid unitree G1 robot to grab an apple with its right arm and place it in a bowl to the side

**Supported Robots: Unitree G1**

**Randomizations:**
- the bowl's xy position is randomized on top of a table in the region [0.025, 0.025] x [-0.025, -0.025]. It is placed flat on the table
- the apple's xy position is randomized on top of a table in the region [0.025, 0.025] x [-0.025, -0.025]. It is placed flat on the table
- the apple's z-axis rotation is randomized to a random angle

**Success Conditions:**
- the apple position is within 0.05m euclidean distance of the bowl's position.
- the robot's right hand is kept outside the bowl and is above it by at least 0.125m.

**Goal Specification:**
- The bowl's 3D position
:::

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/UnitreeG1PlaceAppleInBowl-v1_rt.mp4" type="video/mp4">
</video>
