# ManiSkill1 Environments

We migrate 4 [ManiSkill1](https://github.com/haosulab/ManiSkill) environments to ManiSkill2.

The major changes that can change behaviors are listed as follows

- Assets are refined with an advanced convex decomposition method [CoACD](https://github.com/SarahWeiii/CoACD).
- The robot URDF is updated with official mass and inertia, as well as finger collisions.
- The success metric does not depend on multiple timesteps.
- The GT segmentation masks are removed from observations.
- If the model does not change, the simulation scene will be reused when the env is reset.
