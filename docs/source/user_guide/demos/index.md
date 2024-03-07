# ManiSkill Demos

## Demo Random Actions

`python -m mani_skill2.examples.demo_random_action` lets you run a simulation with random actions. 

Run 

```bash
python -m mani_skill2.examples.demo_random_action -h
```
for a full list of available commands

Some recommended examples that cover a number of features of ManiSkill

Tasks in Realistic Scenes (ReplicaCAD dataset example)
```bash
python -m mani_skill2.utils.download_asset "ReplicaCAD"
python -m mani_skill2.examples.demo_random_action -e "ReplicaCAD_SceneManipulation-v1" \
  --render-mode="rgb_array" --record-dir="videos" # run headless and save video
python -m mani_skill2.examples.demo_random_action -e "ReplicaCAD_SceneManipulation-v1" \
  --render-mode="human" # run with GUI
```

To turn ray-tracing on for more photo-realistic rendering, you can add `--shader="rt"` or `--shader="rt-fast`

```bash
python -m mani_skill2.examples.demo_random_action -e "ReplicaCAD_SceneManipulation-v1" \
  --render-mode="human" --shader="rt-fast" # faster ray-tracing option but lower quality
```

<video preload="auto" controls="True" width="100%">
<source src="/_static/videos/fetch_random_action_replica_cad_rt.mp4" type="video/mp4">
</video>

<!-- 
AI2THOR related scenes
```bash
python -m mani_skill2.utils.download_asset "AI2THOR"
python -m mani_skill2.examples.demo_random_action -e "ArchitecTHOR_SceneManipulation-v1" --render-mode="rgb_array" --record-dir="videos" # run headless and save video
python -m mani_skill2.examples.demo_random_action -e "ArchitecTHOR_SceneManipulation-v1" --render-mode="human" # run with GUI
``` -->

## GPU Simulation

To benchmark the GPU simulation on the PickCube-v1 environment with 4096 parallel environments you can run
```bash
python -m mani_skill2.examples.benchmarking.gpu_sim -e "PickCube-v1" -n 4096
```

To save videos of the visual observations the agent would get (in this case it is just rgb and depth) you can run
```bash
python -m mani_skill2.examples.benchmarking.gpu_sim -e "PickCube-v1" -n 64 --save-video --render-mode="sensors"
```
it should run quite fast! (3000+ fps on a 4090, you can increase the number envs for higher FPS)

More details and performance benchmarking results can be found on [this page](../additional_resources/performance_benchmarking.md)

## Interactive Control

Click+Drag Teleoperation:

Simple tool where you can click and drag the end-effector of the Panda robot arm to solve various tasks. You just click+drag, press "n" to move to the position you dragged to, "g" to toggle on/off grasping, and repeat. Press "q" to quit and save a video of the result.

```bash
python -m mani_skill2.examples.teleoperation.interactive_panda -e "StackCube-v1" 
```

See [main page](../data_collection/teleoperation.md#clickdrag-system) for more details about how to use this tool (for demo and data collection). The video below shows the system running.

<video preload="auto" controls="True" width="100%">
<source src="/_static/videos/teleop-stackcube-demo.mp4" type="video/mp4">
</video>