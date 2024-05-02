# Control Tasks
[asset-badge]: https://img.shields.io/badge/download%20asset-yes-blue.svg
[reward-badge]: https://img.shields.io/badge/dense%20reward-yes-green.svg

These tasks are typical classical control tasks, some of which is adapted from the [DM-Control Suite](https://github.com/google-deepmind/dm_control/). Some tasks directly import the MJCF files from DM-Control.

## MS-CartpoleBalance-v1
![dense-reward][reward-badge]

:::{dropdown} Task Card
:icon: note
:color: primary

**Task Description:**
Use the Cartpole robot to balance a pole on a cart.


**Supported Robots: Cartpole**

**Randomizations:**
- Pole direction is randomized around the vertical axis. the range is [-0.05, 0.05] radians.

**Fail Conditions:**
- Pole is lower than the horizontal plane
:::
<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/MS-CartpoleBalance-v1_rt.mp4" type="video/mp4">
</video>

## MS-CartpoleSwingup-v1
![dense-reward][reward-badge]

:::{dropdown} Task Card
:icon: note
:color: primary

**Task Description:**
Use the Cartpole robot to swing up a pole on a cart.


**Supported Robots: Cartpole**

**Randomizations:**
- Pole direction is randomized around the whole circle. the range is [-pi, pi] radians.

**Success Conditions:**
- No specific success conditions. The task is considered successful if the pole is upright for the whole episode. We can threshold the episode accumulated reward to determine success.
:::