# Digital Twins

ManiSkill supports both training and evaluation types of digital twins and provides a simple framework for building them. Training digital twins are tasks designed to train a robot in simulation to then be deployed in the real world (sim2real). Evaluation digital twins are tasks designed to evaluate the performance of a robot trained on real world data (real2sim) and not for training.


## Training Digital Twins (WIP)

Coming soon.

## BridgeData v2 (Evaluation)

We currently support evaluation digital twins of some tasks in the [BridgeData v2](https://rail-berkeley.github.io/bridgedata/) environments in simulation based on [SimplerEnv](https://simpler-env.github.io/) by Xuanlin Li, Kyle Hsu, Jiayuan Gu et al. These digital twins are also GPU parallelized enabling large-scale, fast, evaluation of real-world generalist robotics policies. GPU simulation + rendering enables evaluating up to 60x faster than the real-world and 10x faster than CPU simulation, all without human supervision. ManiSkill only provides the environments, to run policy inference of models like Octo and RT see https://github.com/simpler-env/SimplerEnv/tree/maniskill3

If you use the BridgeData v2 digital twins please cite the following in addition to ManiSkill 3:

```
@article{li24simpler,
  title={Evaluating Real-World Robot Manipulation Policies in Simulation},
  author={Xuanlin Li and Kyle Hsu and Jiayuan Gu and Karl Pertsch and Oier Mees and Homer Rich Walke and Chuyuan Fu and Ishikaa Lunawat and Isabel Sieh and Sean Kirmani and Sergey Levine and Jiajun Wu and Chelsea Finn and Hao Su and Quan Vuong and Ted Xiao},
  journal = {arXiv preprint arXiv:2405.05941},
  year={2024},
} 
```

### PutCarrotOnPlateInScene-v1

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/digital_twins/bridge_data_v2/PutCarrotOnPlateInScene-v1.mp4" type="video/mp4">
</video>

### PutSpoonOnTableClothInScene-v1

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/digital_twins/bridge_data_v2/PutSpoonOnTableClothInScene-v1.mp4" type="video/mp4">
</video>

### StackGreenCubeOnYellowCubeBakedTexInScene-v1

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/digital_twins/bridge_data_v2/StackGreenCubeOnYellowCubeBakedTexInScene-v1.mp4" type="video/mp4">
</video>

### PutEggplantInBasketScene-v1

<video preload="auto" controls="True" width="100%">
<source src="https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/digital_twins/bridge_data_v2/PutEggplantInBasketScene-v1.mp4" type="video/mp4">
</video>