<!-- THIS IS ALL GENERATED DOCUMENTATION via generate_robot_docs.py. DO NOT MODIFY THIS FILE DIRECTLY. -->

# Robots
<img src="/_static/robot_images/robot-grid.png" alt="Robot Grid" style="width: 100%; height: auto;">


This sections here show the already built/modelled robots ready for simulation across a number of different categories. Some of them are displayed above in an empty environment using a predefined keyframe. Note that not all of these robots are used in tasks in ManiSkill, and some are not tuned for maximum efficiency yet or for sim2real transfer. You can generally assume robots that are used in existing tasks in ManiSkill are of the highest quality and already tuned.

To learn about how to load your own custom robots see [the tutorial](../user_guide/tutorials/custom_robots.md).

## Robots Table
Table of all robots modelled in ManiSkill. Click the robot's picture to see more details on the robot, including more views, collision models, controllers implemented and more.

A quality rating is also given for each robot which rates the robot on how well modelled it is. It follows the same scale as [Mujoco Menagerie](https://github.com/google-deepmind/mujoco_menagerie?tab=readme-ov-file#model-quality-and-contributing)

| Grade | Description                                                 |
|-------|-------------------------------------------------------------|
| A+    | Values are the product of proper system identification      |
| A     | Values are realistic, but have not been properly identified |
| B     | Stable, but some values are unrealistic                     |
| C     | Conditionally stable, can be significantly improved         |

Robots that are cannot be stably simulated are not included in ManiSkill at all. Most robots will have a grade of B (essentially does it look normal in simulation). While some robots may have grades of A/A+ we still strongly recommend you perform your own system ID as each robot might be a bit different.

<div class="gallery" style="display: flex; flex-wrap: wrap; gap: 10px;">

</div>

```{toctree}
:caption: Directory
:maxdepth: 1

anymal_c/index
allegro_hand_left/index
allegro_hand_right/index
allegro_hand_right_touch/index
dclaw/index
fetch/index
floating_inspire_hand_right/index
floating_panda_gripper/index
floating_robotiq_2f_85_gripper/index
googlerobot/index
humanoid/index
panda/index
panda_stick/index
panda_wristcam/index
stompy/index
trifingerpro/index
ur_10e/index
unitree_g1/index
unitree_g1_simplified_legs/index
unitree_g1_simplified_upper_body/index
unitree_go2/index
unitree_h1/index
unitree_h1_simplified/index
widowx250s/index
xarm6_nogripper/index
xarm6_robotiq/index
xarm6_robotiq_wristcam/index
xarm7_ability/index

```