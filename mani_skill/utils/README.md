# ManiSkill Utilities

These are various functions/tooling that help make it easy for ManiSkill to work, as well as build your own tasks, train on them, and evaluate on them.

Description of the main modules are as follows:

`building/` - All useful utility code for building a task and/or scene. Includes functions for loading assets and articulations from various datasets, and randomization functions useful for randomizing task initialization.
`scene_builder/` - Contains code relating to the `SceneBuilder` class and provides some prebuilt scene builders for a standard table top scene, [ReplicaCAD](https://aihabitat.org/datasets/replica_cad/), as well as scenes from [AI2THOR](https://ai2thor.allenai.org/) via the [HSSD dataset](https://huggingface.co/datasets/hssd/ai2thor-hab).

`geometry/` - Various functions for working with geometry, from sampling primitive shapes to getting axis-aligned bounding boxes of articulations/actors.

`wrappers/` - Wrapper classes that provide additional functionality such as recording videos/episodes, modifying observation spaces, as well as adapting the environment API so that RL libraries such as [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) work out of the box.

`visualization/` - Visualization tools

`gym_utils.py` - Various utilities for working with the gymnasium/gym API

`common.py` - A ton of fairly common utilities, including those that are often used for reward functions, success evaluation, as well as working with nested dictionaries.