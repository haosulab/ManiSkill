# Teleoperation

There are a number of teleoperation systems provided by ManiSkill that help collect demonstration data in environments. Each system is detailed below with how to use it and a demo video. We also detail what hardware requirements are necessary, how usable the system is, and the limitations of the system.

At the moment there is the intuitive click+drag system, systems using e.g. space mouse, a VR headset will come soon.

## Click+Drag System

Requirements: Display, mouse, keyboard

Usability: Extremely easy to generate fine-grained demonstrations

Limitations: Limited to only solving less dynamical tasks with two-finger grippers like picking up a cube. Tasks like throwing a cube would not be possible.

To start the system you can specify an environment id with `-e` and run
```bash
python -m mani_skill2.examples.interactive_teleop.py -e "PickCube-v1"
```

You can then drag the end-effector of the robot arm around to any position and rotation and press "n" on the keyboard to generate a trajectory to that place (done via motion planning). Each time the system will also print the current info about whether the task is solved or not.

You can press "g" to toggle the gripper to be closing or opening.

To finish collecting one trajectory and to move on to another, simply press "c" which will save the last trajectory.

To stop data collection press "q" to quit.

You can always press "h" to bring up a help menu describing the keyboard commands

<!-- TODO (stao): discuss checkpointing method, help button -->

<!-- ## Space Mouse -->
<!-- 
## Meta Quest 3

Requirements: Meta Quest 3 -->