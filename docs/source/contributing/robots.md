# Robots

To add a new robot, you can follow any of the existing robots built already as templates in ManiSkill. We also highly recommend that you read through [custom robot tutorial](../user_guide/tutorials/custom_robots.md) to learn how to make new robots and tune them.

ManiSkill is a supporter of open-source and we encourage you to make contributions to help grow our list of robots in simulation!

## Contributing the Robot to ManiSkill

We recommend first opening an issue on our GitHub about your interest in adding a new robot as to not conflict with others and to consolidate information. Once done our maintainers can give a go ahead.

In your pull request, we ask you to do the following:
- The robot / agent class code should be placed in `mani_skill/agents/<agent_group_name>/your_robot.py`. If you want to re-use an agent class (e.g. as done with the Allegro hand robot and the Panda robot) you can create a folder that groups all the different agent classes together.
- Update `mani_skill/utils/download_asset.py` and add a data source linking to your new robot under the `initialize_extra_resources` function. We recommend you to comment on the issue you opened with the asset files so we can then create a repository for it that is versioned. You can also create a repository yourself, you can follow how https://github.com/haosulab/ManiSkill-ANYmalC/ is setup to do this. The easiest way to allow access to assets is to create a Github release and link the source code zip file in the `download_asset.py` code
- Finally document the new robot! Add an appropriate section to `docs/source/robots` and follow the template of the other robots there.




There should be some automated test suites coming soon but for now, maintainers will test the robot manually to see if it generally works.
