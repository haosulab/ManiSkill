# Quadruped Tasks

[asset-badge]: https://img.shields.io/badge/download%20asset-yes-blue.svg
[reward-badge]: https://img.shields.io/badge/dense%20reward-yes-green.svg

We have provided some quadruped related tasks, but are still in the process of optimizing the simulation speed and providing better reward functions. We welcome any open source contributions to these tasks!

## AnymalC-Reach-v1
![dense-reward][reward-badge]

:::{dropdown} Task Card
:icon: note
:color: primary

**Task Description:**
Control the AnymalC robot to reach a target location in front of it. Note the current reward function works but more needs to be added to constrain the learned quadruped gait looks more natural

**Supported Robots: AnymalC**

**Randomizations:**
- Robot is initialized in a stable rest/standing position
- The goal for the robot to reach is initialized 2.5 +/- 0.5 meters in front, and +/- 1 meters to either side

**Success Conditions:**
- If the robot position is within 0.35 meters of the goal

**Fail Conditions:**
- If the robot has fallen over, which is considered True when the main body (the center part) hits the ground

**Goal Specification:**
- The 2D goal position in the XY-plane
:::

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/AnymalC-Reach-v1_rt.mp4" type="video/mp4">
</video>

## AnymalC-Spin-v1
![dense-reward][reward-badge]

:::{dropdown} Task Card
:icon: note
:color: primary

**Task Description:**
Control the AnymalC robot to spin around in place as fast as possible and is rewarded by its angular velocity.

**Supported Robots: AnymalC**

**Randomizations:**
- Robot is initialized in a stable rest/standing position

**Fail Conditions:**
- If the robot has fallen over, which is considered True when the main body (the center part) hits the ground

:::

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/AnymalC-Spin-v1_rt.mp4" type="video/mp4">
</video>