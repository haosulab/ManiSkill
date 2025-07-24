# SO100

This is the 6DOF SO100 arm. The original assets are from [here](https://github.com/TheRobotStudio/SO-ARM100) and were provided under a [Apache 2.0 License](LICENSE)

Changes made:
- Fixed joint limits to reflect real world behavior
- Fixed joint tags from continuous to revolute which permit joint limits
- Fixed joint directions and orientations to match the real robot's joints
- removed spaces in link names
- manual decomposition of gripper link collision meshes into simpler meshes
