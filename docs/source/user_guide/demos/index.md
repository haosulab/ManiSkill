# {material-regular}`notes` Demos

There are a number of useful/quick scripts you can run to do a quick test/demonstration of various features of ManiSkill.

## Demo Random Actions
The fastest and quickest demo is the random actions demo.
Run 
```bash
python -m mani_skill.examples.demo_random_action -h
```
for a full list of available commands.

Some recommended examples that cover a number of features of ManiSkill

Tasks in Realistic Scenes (ReplicaCAD dataset example)
```bash
python -m mani_skill.utils.download_asset "ReplicaCAD"
python -m mani_skill.examples.demo_random_action -e "ReplicaCAD_SceneManipulation-v1" \
  --render-mode="rgb_array" --record-dir="videos" # run headless and save video
python -m mani_skill.examples.demo_random_action -e "ReplicaCAD_SceneManipulation-v1" \
  --render-mode="human" # run with GUI
```

To turn ray-tracing on for more photo-realistic rendering, you can add `--shader="rt"` or `--shader="rt-fast`

```bash
python -m mani_skill.examples.demo_random_action -e "ReplicaCAD_SceneManipulation-v1" \
  --render-mode="human" --shader="rt-fast" # faster ray-tracing option but lower quality
```

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/docs/source/_static/videos/fetch_random_action_replica_cad_rt.mp4" type="video/mp4">
</video>

Tasks with multiple robots
```bash
python -m mani_skill.examples.demo_random_action -e "TwoRobotStackCube-v1" \
  --render-mode="human"
```

```{figure} images/tworobotstackcube.png
---
alt: SAPIEN GUI two robot stack cube task
---
```


Tasks with dextrous hand
```bash
python -m mani_skill.examples.demo_random_action -e "RotateValveLevel2-v1" \
  --render-mode="human"
```

```{figure} images/rotatevalvelevel2.png
---
alt: SAPIEN GUI showing the rotate valve level 2 task
---
```


Tasks with simulated tactile sensing
```bash
python -m mani_skill.examples.demo_random_action -e "RotateSingleObjectInHandLevel3-v1" \
  --render-mode="human"
```

```{figure} images/rotatesingleobjectinhand.png
---
alt: SAPIEN GUI showing the rotate single object in hand task
---
```

This task also uses a feature unique to ManiSkill/SAPIEN where you can retrieve object-pair contact impulses/forces, in addition to object net contact forces. 

To quickly demo tasks that support simulating different objects and articulations (with different dofs) across parallel environments see the [GPU Simulation section](#gpu-simulation)

<!-- 
AI2THOR related scenes
```bash
python -m mani_skill.utils.download_asset "AI2THOR"
python -m mani_skill.examples.demo_random_action -e "ArchitecTHOR_SceneManipulation-v1" --render-mode="rgb_array" --record-dir="videos" # run headless and save video
python -m mani_skill.examples.demo_random_action -e "ArchitecTHOR_SceneManipulation-v1" --render-mode="human" # run with GUI
``` -->

## GPU Simulation

To benchmark the GPU simulation on the PickCube-v1 task with 4096 parallel tasks you can run
```bash
python -m mani_skill.examples.benchmarking.gpu_sim -e "PickCube-v1" -n 4096
```

To save videos of the visual observations the agent would get (in this case it is just rgb and depth) you can run
```bash
python -m mani_skill.examples.benchmarking.gpu_sim -e "PickCube-v1" -n 64 --save-video --render-mode="sensors"
```
it should run quite fast! (3000+ fps on a 4090, you can increase the number envs for higher FPS). You can change `--render-mode="rgb_array"` to render from higher quality cameras.

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/docs/source/_static/videos/mani_skill_gpu_sim-PickCube-v1-num_envs=16-obs_mode=state-render_mode=sensors.mp4" type="video/mp4">
</video>


To try out the heterogenous object simulation features you can run
```bash
python -m mani_skill.examples.benchmarking.gpu_sim -e "PickSingleYCB-v1" -n 64 --save-video --render-mode="sensors"
python -m mani_skill.examples.benchmarking.gpu_sim -e "RotateValveLevel2-v1" -n 64 --save-video --render-mode="sensors"
```
which shows two tasks that have different objects and articulations in every parallel environment. Below is an example for the PickSingleYCB task.

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/docs/source/_static/videos/mani_skill_gpu_sim-PickSingleYCB-v1-num_envs=16-obs_mode=state-render_mode=sensors.mp4" type="video/mp4">
</video>


<!-- TODO show mobile manipulation scene gpu sim stuff -->

More details and performance benchmarking results can be found on [this page](../additional_resources/performance_benchmarking.md)

## Interactive Control

Click+Drag Teleoperation:

Simple tool where you can click and drag the end-effector of the Panda robot arm to solve various tasks. You just click+drag, press "n" to move to the position you dragged to, "g" to toggle on/off grasping, and repeat. Press "q" to quit and save a video of the result.

```bash
python -m mani_skill.examples.teleoperation.interactive_panda -e "StackCube-v1" 
```

See [main page](../data_collection/teleoperation.md#clickdrag-system) for more details about how to use this tool (for demo and data collection). The video below shows the system running.

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/docs/source/_static/videos/teleop-stackcube-demo.mp4" type="video/mp4">
</video>

## Motion Planning Solutions

We provide some motion planning solutions/demos for the panda arm on some tasks, you can try it now and record demonstrations with the following:

```bash
python -m mani_skill.examples.motionplanning.panda.run -e "PickCube-v1" # runs headless and only saves video
python -m mani_skill.examples.motionplanning.panda.run -e "StackCube-v1" --vis # opens up the GUI
python -m mani_skill.examples.motionplanning.panda.run -h # open up a help menu and also show what tasks have solutions
```

Example below shows what it looks like with the GUI:

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/docs/source/_static/videos/motionplanning-stackcube.mp4" type="video/mp4">
</video>

For more details check out the [motion planning page](../data_collection/motionplanning.md)

## Visualize Pointcloud Data

You can run the following to visualize the pointcloud observations (require's a display to work)

```bash
pip install "pyglet<2" # make sure to install this dependency
python -m mani_skill.examples.demo_vis_pcd -e "StackCube-v1"
```


```{figure}  images/pcd_vis.png
```