# Scene Datasets

We provide a command line tool to download scene datasets (typically adapted from the original dataset). 

We can support loading just about any scene dataset but currently only provide an option to load ReplicaCAD as it is the best tested one (with interactable objects) we have access to. We are in the process still of processing AI2THOR scene datasets to make them available as well in ManiSkill and standardizing a Scene-level task / definition API to make it easier to support running GPU parallelized simulation in any scene

```bash
# list all scene datasets available for download
python -m mani_skill.utils.download_asset --list "scene"
python -m mani_skill.utils.download_asset ReplicaCAD
```
