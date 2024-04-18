# Adding Robots

This is currently a WIP

<!-- TODO: Detail how to add and model a robot to run in ManiSkill/SAPIEN.
- Cover working with URDFs, fixing common URDF issues
- Cover certain disabling collisions for efficiency
- Cover how to choose drive properties, how to determine when to create drive, tendons etc... -->

## Importing from URDF

For example code we recommend checking out https://github.com/haosulab/ManiSkill2/blob/dev/mani_skill/agents/robots/panda/panda.py

For a starting template check out https://github.com/haosulab/ManiSkill2/blob/dev/mani_skill/agents/robots/_template/template_robot.py

## Importing from Mujoco MJCF

ManiSkill supports importing [Mujoco's MJCF format](https://mujoco.readthedocs.io/en/latest/modeling.html) of files to load robots (and other objects), although not all features are supported.

For example code that loads the robot and the scene see https://github.com/haosulab/ManiSkill2/blob/dev/mani_skill/envs/tasks/control/cartpole.py


At the moment, the following are not supported:
- Procedural texture generation
- Importing motors and solver configurations
- Correct use of contype and conaffinity attributes (contype of 0 means no collision mesh is added, otherwise it is always added)
- Correct use of groups (at the moment anything in group 0 and 2 can be seen, other groups are hidden all the time)
