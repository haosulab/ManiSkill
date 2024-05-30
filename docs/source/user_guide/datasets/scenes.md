# Scene Datasets

We provide a command line tool to download scene datasets (typically adapted from the original dataset). 

ManiSkill can build any scene provided assets are provided. ManiSkill out of the box provides code and download links to use the [ReplicaCAD](https://aihabitat.org/datasets/replica_cad/) and [AI2THOR](https://github.com/allenai/ai2thor) set of scenes (shown below). These are picked because we are able to make them *interactive* scenes where objects can be manipulated and moved around.

```{figure} images/two_scenes_examples.png
```

```bash
# list all scene datasets available for download
python -m mani_skill.utils.download_asset --list "scene"
python -m mani_skill.utils.download_asset ReplicaCAD # small scene and fast to download
python -m mani_skill.utils.download_asset AI2THOR # lots of scenes and slow to download
```

## Exploring the Scene Datasets

To explore the scene datasets, you can provide an environment ID and a seed (to change which scene is sampled if there are several available) and run the random action script. Shown below are the two environment IDs configured already to enable you to play with ReplicaCAD and ArchitecTHOR, one of the scene sets in AI2THOR.

```bash
python -m mani_skill.examples.demo_random_action \
  -e "ReplicaCAD_SceneManipulation-v1" \
  --render-mode="rgb_array" --record-dir="videos" # run headless and save video

python -m mani_skill.examples.demo_random_action \
  -e "ArchitecTHOR_SceneManipulation-v1" --render-mode="human" \
  -s 3 # open a GUI and sample a scene with seed 3
```

## Training on the Scene Datasets

Large scene datasets with hundreds of objects like ReplicaCAD and AI2THOR can be used to train more general purpose robots/agents and also serve as synthetic data generation sources. We are still in the process of providing more example code and documentation about how to best leverage these scene datasets but for now we provide code to explore and interact with the scene datasets.

### Reinforcement Learning / Imitation Learning

We are currently in the process of building task code similar to the ReplicaCAD Rearrange challenge and will open source that when it is complete. Otherwise at the moment there are not any trainable tasks with defined success/fail conditions and/or rewards that use any of the big scene datasets.

### Computer Vision / Synthetic 2D/3D Data Generation (WIP)
