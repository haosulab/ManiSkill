# Motion Planning

ManiSkill provides simple tools to use motion planning to generate robot trajectories, primarily via the open-source [mplib](https://github.com/haosulab/MPlib) library. If you install ManiSkill, mplib will come installed already so no extra installation is necessary.

For an in depth tutorial on how to use more advanced features of mplib check out their documentation here: https://motion-planning-lib.readthedocs.io/latest/. Otherwise this section will cover some example code you can use and modify to generate motion planned demonstrations. The example code here is written for the Panda arm but should be modifiable to work for other robots.

## Motion Planning with Panda Arm

We provide some built in motion planning solutions for some tasks using the Panda arm at https://github.com/haosulab/ManiSkill2/tree/dev/mani_skill/examples/motionplanning/panda. You can run a quick demo below:

```bash
python -m mani_skill.examples.motionplanning.panda.run -e "PickCube-v1" # runs headless and only saves video
python -m mani_skill.examples.motionplanning.panda.run -e "StackCube-v1" --visualize # opens up the GUI
python -m mani_skill.examples.motionplanning.panda.run -h # open up a help menu and also show what tasks have solutions
```

Example below shows what it looks like with the GUI:

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill2/raw/dev/docs/source/_static/videos/motionplanning-stackcube.mp4" type="video/mp4">
</video>