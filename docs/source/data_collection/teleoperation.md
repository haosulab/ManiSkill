# Teleoperation

There are a number of teleoperation systems provided by ManiSkill that help collect demonstration data in environments. Each system is detailed below with how to use it and a demo video. We also detail what hardware requirements are necessary, how usable the system is, and the limitations of the system

## Click+Drag System

Requirements: Display, mouse, keyboard

Usability: Extremely easy to generate fine-grained demonstrations

Limitations: Limited to only solving less dynamical tasks like picking up a cube. Tasks like throwing a cube would not be possible.

To start the system run
```bash
python -m mani_skill2.examples.interactive_teleop.py -e "PickCube-v1"
```




## Space Mouse

## Meta Quest 3

Requirements: Meta Quest 3