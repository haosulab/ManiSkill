# Humanoids

## Unitree H1

```{figure} ../images/unitree_h1.png
```

Robot UID: `unitree_h1, unitree_h1_simplified`

Agent Class Code: [https://github.com/haosulab/ManiSkill/blob/main/mani_skill/agents/robots/unitree_h1/h1.py](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/agents/robots/unitree_h1/h1.py)

Robot Description: [https://github.com/haosulab/ManiSkill-UnitreeH1](https://github.com/haosulab/ManiSkill-UnitreeH1)

Robot `unitree_h1` comes with a complete set of collision meshes and is not simplified. `unitree_h1_simplified` is a version where most collision meshes are removed and a minimal set is kept for use and faster simulation.

### With Dextrous Hands (WIP)

### With Mounted Sensors (WIP)

Main blocker here is actually its unclear where people typically mount cameras/depth sensors onto Unitree H1

## Unitree G1

```{figure} ../images/unitree_g1.png
```

Robot UID: `unitree_g1, unitree_g1_simplified_legs`

Agent Class Code: [https://github.com/haosulab/ManiSkill/blob/main/mani_skill/agents/robots/unitree_g1/g1.py](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/agents/robots/unitree_g1/g1.py)

Robot Description: [https://github.com/haosulab/ManiSkill-UnitreeG1](https://github.com/haosulab/ManiSkill-UnitreeG1)

Robot `unitree_g1` comes with a complete set of collision meshes and is not simplified. `unitree_g1_simplified_legs` is a version where all collision meshes except those around the legs and feet are removed, useful for fast training without manipulation.


## Stompy (from KScale Labs)

```{figure} ../images/stompy.png
```

Robot UID: `stompy`

Agent Class Code: [https://github.com/haosulab/ManiSkill/blob/main/mani_skill/agents/robots/stompy/stompy.py](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/agents/robots/stompy/stompy.py)

Robot Description: [https://github.com/haosulab/ManiSkill-Stompy](https://github.com/haosulab/ManiSkill-Stompy)

## Mujoco Humanoid

```{figure} ../images/mujoco_humanoid.png
```

Robot UID: `humanoid`

Agent Class Code: [https://github.com/haosulab/ManiSkill/blob/main/mani_skill/agents/robots/humanoid/humanoid.py](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/agents/robots/humanoid/humanoid.py)

Robot Description: [https://github.com/haosulab/ManiSkill/blob/main/mani_skill/assets/robots/humanoid](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/assets/robots/humanoid)