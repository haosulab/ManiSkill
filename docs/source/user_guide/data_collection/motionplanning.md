# Motion Planning

ManiSkill provides simple tools to use motion planning to generate robot trajectories, primarily via the open-source [mplib](https://github.com/haosulab/MPlib) library. If you install ManiSkill, mplib will come installed already so no extra installation is necessary.

For an in-depth tutorial on how to use more advanced features of mplib check out their documentation here: https://motion-planning-lib.readthedocs.io/latest/. Otherwise this section will cover some example code you can use and modify to generate motion planned demonstrations. The example code here is written for the Panda arm but can be modified to work for other robots.

## Motion Planning with Panda Arm

We provide some built-in motion planning solutions for some tasks using the Panda arm at https://github.com/haosulab/ManiSkill/tree/main/mani_skill/examples/motionplanning/panda. You can run a quick demo below, which will save trajectory data as .h5 files to `demos/motionplanning/<env_id>` and optionally save videos and/or visualize with a GUI.

```bash
python -m mani_skill.examples.motionplanning.panda.run -e "PickCube-v1" --save-video # runs headless and only saves video
python -m mani_skill.examples.motionplanning.panda.run -e "StackCube-v1" --vis # opens up the GUI
python -m mani_skill.examples.motionplanning.panda.run -h # open up a help menu and also show what tasks have solutions
```

Example below shows what it looks like with the GUI:

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/docs/source/_static/videos/motionplanning-stackcube.mp4" type="video/mp4">
</video>

The solutions to these tasks usually involve decomposing the task down to a sequence of simple pick, place, and movements. The example code provided controls the panda arm's end-effector to move to any pose in its workspace as well as grab/release.

For example, the PickCube-v1 task is composed of
1. move gripper just over the red cube and orient the gripper so it faces the same direction as the cube
2. move gripper down so the fingers surround the cube
3. close the gripper
4. move the gripper to above the goal location so the tool center point (tcp) of the gripper is at the goal

Note that while motion planning can generate and solve a wide variety of tasks, its main limitation is that it often requires an human/engineer to tune and write, as well as being unable to generate solutions for more dynamical tasks.