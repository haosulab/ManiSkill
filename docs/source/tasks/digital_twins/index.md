# Digital Twins (WIP)

## PushT-easy-v1
![dense-reward][reward-badge]
:::{dropdown} Task Card
:icon: note
:color: primary

**Task Description:**
Easier Digital Twin of real life push-T task from Diffusion Policy: https://diffusion-policy.cs.columbia.edu/
'In this task, the robot needs to
① precisely push the T- shaped block into the target region, and
② move the end-effector to the end-zone which terminates the episode.' [② Not required for PushT-easy-v1]

**Supported Robots: PandaStick**

**Randomizations:**
- The 3D T block's initial center of mass is randomized in the region on the table: [-1,2] x [-1,1] + T Goal initial position. It is placed flat on the table
- The 3D T block's initial z rotation is randomized in [0,2pi] around the center of mass of the block

**Success Conditions:**
- the 3D T block covers at least 90% of the 2D T goal zone
:::

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PushCube-v1_rt.mp4" type="video/mp4">
</video>