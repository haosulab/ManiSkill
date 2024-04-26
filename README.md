# ManiSkill

**ManiSkill 3 beta version releases soon. We strongly recommend new users to wait until that releases. To stay tuned you can star/watch this repository and/or join our [discord](https://discord.gg/x8yUZe5AdN) and/or follow our twitter https://twitter.com/HaoSuLabUCSD. For now you can read about the features below.**

**Users looking for the original ManiSkill2 can find the commit for that codebase at https://github.com/haosulab/ManiSkill/tree/v0.5.3**




![teaser](figures/teaser_v2.jpg)

<!-- [![PyPI version](https://badge.fury.io/py/mani-skill.svg)](https://badge.fury.io/py/mani-skill) -->
<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haosulab/ManiSkill/blob/main/examples/tutorials/1_quickstart.ipynb) -->
[![Docs status](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://maniskill.readthedocs.io/en/latest/)
[![Discord](https://img.shields.io/discord/996566046414753822?logo=discord)](https://discord.gg/x8yUZe5AdN)

ManiSkill is a powerful unified framework for robot simulation and training powered by [SAPIEN](https://sapien.ucsd.edu/). The entire stack is as open-source as possible. Among its features, it includes
- GPU parallelized visual data collection system. A policy can collect RGBD + Segmentation data at about 10,000+ FPS with a 4090 GPU, 10-100x faster compared to simulators like Isaac Sim.
- Example tasks covering a wide range of different robot embodiments (quadruped, mobile manipulators, single-arm robots) as well as a wide range of different tasks (table-top, locomotion, scene-level manipulation)
- GPU parallelized tasks, enabling incredibly fast synthetic data collection in simulation at the same or faster speed as other GPU sims like IsaacSim
- GPU parallelized tasks support simulating diverse scenes where every parallel environment has a completely different scene/set of objects
- Flexible task building API that abstracts away much of the complex GPU memory management code
<!-- - Evaluate models trained on real-world data in simulation, no robot hardware needed -->

ManiSkill plans to enable all kinds of workflows, including but not limited to 2D/3D vision-based reinforcement learning, imitation learning, sense-plan-act, etc. At the moment the repo contains just state/visual RL baselines, others are coming soon.
<!-- 
Please refer to our [documentation](https://maniskill.readthedocs.io/en/main/) to learn more information from tutorials on building tasks to data collection. To quickly get started after installation check out https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html.


Note previously there was previously a ManiSkill and ManiSkill2, we are rebranding it all to just ManiSkill and the python package versioning tells you which iteration (3.0.0 now means ManiSkill3)


**Table of Contents**

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Demonstrations](#demonstrations)
- [Leaderboard](#leaderboard)
- [License](#license)
- [Citation](#citation)

## Installation

Installation of ManiSkill is extremely simple, you only need to run a few pip installs

```bash
# install the package
pip install --upgrade mani_skill
# install a version of torch that is compatible with your system
pip install torch torchvision torchaudio
```

You can also install the main `mani_skill` package from github/source:

```bash
# GitHub
pip install --upgrade git+https://github.com/haosulab/ManiSkill.git

# Source
git clone https://github.com/haosulab/ManiSkill.git
cd ManiSkill && pip install -e .

# remember to install a version of torch that is compatible with your system
pip install torch torchvision torchaudio
```

---

A GPU with the Vulkan driver installed is required to enable rendering in ManiSkill. See [these docs](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan) for instructions to fix Vulkan related bugs. The rigid-body environments, powered by SAPIEN, are ready to use after installation. Test your installation:

```bash
# Run an episode of "PickCube-v1" (a rigid-body environment) with random actions
# Or specify an environment by "-e ${ENV_ID}"
python -m mani_skill.examples.demo_random_action
```

For a full list of example scripts you can run see [the docs](https://maniskill.readthedocs.io/en/latest/user_guide/demos/index.html).

Some environments require **downloading assets**. You can download download task-specific assets by `python -m mani_skill.utils.download_asset ${ENV_ID}`. The assets will be downloaded to `~/maniskill/data` by default, and you can also use the environment variable `MS_ASSET_DIR` to specify this destination.

Please refer to our [documentation](https://maniskill.readthedocs.io/en/latest) for details on all supported environments. The documentation also indicates which environments require downloading assets.

---

We further provide a docker image (`haosulab/mani-skill`) on [Docker Hub](https://hub.docker.com/repository/docker/haosulab/mani-skill/general) and its corresponding [Dockerfile](./docker/Dockerfile).

If you encounter any issues with installation, please see the [troubleshooting](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html) section for common fixes or submit an [issue](https://github.com/haosulab/ManiSkill/issues).

## Getting Started

To quickly get started check out the quick start page on the docs: https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html

## License

All rigid body environments in ManiSkill are licensed under fully permissive licenses (e.g., Apache-2.0).

The assets are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode).

## Citation

If you use ManiSkill or its assets and models, consider citing the following publication:

```
@inproceedings{gu2023maniskill2,
  title={ManiSkill2: A Unified Benchmark for Generalizable Manipulation Skills},
  author={Gu, Jiayuan and Xiang, Fanbo and Li, Xuanlin and Ling, Zhan and Liu, Xiqiaing and Mu, Tongzhou and Tang, Yihe and Tao, Stone and Wei, Xinyue and Yao, Yunchao and Yuan, Xiaodi and Xie, Pengwei and Huang, Zhiao and Chen, Rui and Su, Hao},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
``` -->
