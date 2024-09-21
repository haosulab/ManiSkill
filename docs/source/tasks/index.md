# Tasks

[asset-badge]: https://img.shields.io/badge/download%20asset-yes-blue.svg
[reward-badge]: https://img.shields.io/badge/dense%20reward-yes-green.svg

ManiSkill features a number of built-in rigid-body tasks, all GPU parallelized and demonstrate a range of features. They are generally categorized into a few categories.

Soft-body tasks will be added back in as they are still in development as part of a new soft-body simulator we are working on. For some categories there are very few tasks and/or no dense rewards as this is the beta release. We are in the process of still adding some examples in (and welcome outside contributions on these efforts!)

For each task documented in these sections we provide a "Task Card" which briefly describes all the important aspects of the task, including task description, supported robots, randomizations, success/fail conditions, and goal specification in observations. We further show tags describing whether there are dense rewards provided, and if assets need to be downloaded

![dense-reward][reward-badge]
![download-asset][asset-badge]

If assets do need to be downloaded you can just run

```python -m mani_skill.utils.download_asset <env_id>```

Note that some tasks have different state observation data provided depending on the observation mode. In general when the observation mode is "state" or "state_dict" then all ground truth data necessary to solve the task is given. If the observation mode is any of the visual ones, we remove any ground truth observation data (like an object's pose) that would normally be unattainable in the real world.

```{toctree}
:maxdepth: 1

control/index
table_top_gripper/index
quadruped/index
humanoid/index
mobile_manipulation/index
dextrous/index
digital_twins/index
drawing/index
```
