# Mobile Manipulation Tasks

[asset-badge]: https://img.shields.io/badge/download%20asset-yes-blue.svg
[reward-badge]: https://img.shields.io/badge/dense%20reward-yes-green.svg

There is currently just the one task, we are still in the process of porting over usable environments from past ManiSkill versions (e.g. Pushing a chair).

## OpenCabinetDrawer-v1
![dense-reward][reward-badge]

:::{dropdown} Task Card
:icon: note
:color: primary

**Task Description:**
Use the Fetch mobile manipulation robot to move towards a target cabinet and open the target drawer out.

**Supported Robots: Fetch**

**Randomizations:**
- Robot is randomly initialized 1.6 to 1.8 meters away from the cabinet and positioned to face it
- Robot's base orientation is randomized by -9 to 9 degrees
- The cabinet selected to manipulate is randomly sampled from all PartnetMobility cabinets that have drawers
- The drawer to open is randomly sampled from all drawers available to open

**Success Conditions:**
- The drawer is open at least 90% of the way, and the angular/linear velocities of the drawer link are small

**Goal Specification:**
- 3D goal position centered at the center of mass of the handle mesh on the drawer to open (also visualized in human renders with a sphere).

:::
<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/OpenCabinetDrawer-v1_rt.mp4" type="video/mp4">
</video>