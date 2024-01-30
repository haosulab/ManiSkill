# ManiSkill Utilities

Description of the main modules are as follows:

`building/` - All useful utility code for building a task and/or scene. Includes functions for loading assets and articulations from various datasets, and randomization functions useful for randomizing task initialization.
`scene_builder/` - Contains code relating to the `SceneBuilder` class and provides some prebuilt scene builders for a standard table top scene, a fully articulated hand-made kitchen scene, as well as scenes from AI2THOR.

`geometry/` - Various functions for working with geometry, from sampling primitive shapes to getting axis-aligned bounding boxes of articulations/actors.

`wrappers/` - Wrapper classes that provide additional functionality such as recording videos/episodes, modifying observation spaces, as well as adapting the environment API so that RL libraries such as [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) work out of the box.

`visualization/` - Visualization tools