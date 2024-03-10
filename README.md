# ManiSkill

![teaser](figures/teaser_v2.jpg)

[![PyPI version](https://badge.fury.io/py/mani-skill.svg)](https://badge.fury.io/py/mani-skill)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haosulab/ManiSkill2/blob/main/examples/tutorials/1_quickstart.ipynb)
[![Docs status](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://haosulab.github.io/ManiSkill2)
[![Discord](https://img.shields.io/discord/996566046414753822?logo=discord)](https://discord.gg/x8yUZe5AdN)
<!-- [![Docs](https://github.com/haosulab/ManiSkill2/actions/workflows/gh-pages.yml/badge.svg)](https://haosulab.github.io/ManiSkill2) -->

ManiSkill is a unified benchmark for learning generalizable robotic manipulation skills powered by [SAPIEN](https://sapien.ucsd.edu/). **It features 20 out-of-box task families with 2000+ diverse object models and 4M+ demonstration frames**. Moreover, it empowers fast visual input learning algorithms so that **a CNN-based policy can collect samples at about 2000 FPS with 1 GPU and 16 processes on a workstation**. The benchmark can be used to study a wide range of algorithms: 2D & 3D vision-based reinforcement learning, imitation learning, sense-plan-act, etc.

Note previously there was previously a ManiSkill and ManiSkill2, we are rebranding it all to just ManiSkill and the python package versioning tells you which iteration (3.0.0 now means ManiSkill3)

Please refer to our [documentation](https://maniskill.readthedocs.io/en/dev/) to learn more information from tutorials on building tasks to data collection. To quickly get started after installation check out https://maniskill.readthedocs.io/en/dev/user_guide/getting_started/quickstart.html.

<!-- There are also hands-on [tutorials](examples/tutorials) (e.g, [quickstart colab tutorial](https://colab.research.google.com/github/haosulab/ManiSkill2/blob/main/examples/tutorials/1_quickstart.ipynb)). -->

**Table of Contents**

- [Installation](#installation)
- [Getting Started](#getting-started)
<!-- - [Reinforcement Learning Example with ManiSkill2-Learn](#reinforcement-learning-example-with-maniskill2-learn) -->
- [Demonstrations](#demonstrations)
- [Leaderboard](#leaderboard)
- [License](#license)
- [Citation](#citation)

## Installation

From pip:

```bash
pip install mani_skill==3.0.0.dev3
```

From github:

```bash
pip install --upgrade git+https://github.com/haosulab/ManiSkill2.git
```

From source:

```bash
git clone https://github.com/haosulab/ManiSkill2.git
cd ManiSkill2 && pip install -e .
```

---

A GPU with the Vulkan driver installed is required to enable rendering in ManiSkill2. The rigid-body environments, powered by SAPIEN, are ready to use after installation. Test your installation:

```bash
# Run an episode (at most 50 steps) of "PickCube-v1" (a rigid-body environment) with random actions
# Or specify an environment by "-e ${ENV_ID}"
python -m mani_skill.examples.demo_random_action
```

Some environments require **downloading assets**. You can download download task-specific assets by `python -m mani_skill.utils.download_asset ${ENV_ID}`. The assets will be downloaded to `~/maniskill/data` by default, and you can also use the environment variable `MS_ASSET_DIR` to specify this destination.

Please refer to our [documentation](https://haosulab.github.io/ManiSkill2/concepts/environments.html) for details on all supported environments. The documentation also indicates which environments require downloading assets.

---

The soft-body environments are based on SAPIEN and customized [NVIDIA Warp](https://github.com/NVIDIA/warp), which requires **CUDA toolkit >= 11.3 and gcc** to compile. Please refer to the [documentation](https://haosulab.github.io/ManiSkill2/getting_started/installation.html#warp-maniskill2-version) for more details about installing ManiSkill2 Warp.

---

We further provide a docker image (`haosulab/mani-skill`) on [Docker Hub](https://hub.docker.com/repository/docker/haosulab/mani-skill/general) and its corresponding [Dockerfile](./docker/Dockerfile).

If you encounter any issues with installation, please see the [troubleshooting](https://maniskill.readthedocs.io/en/dev/user_guide/getting_started/installation.html) section for common fixes or submit an [issue](https://github.com/haosulab/ManiSkill2/issues).

## Getting Started

To quickly get started check out the quick start page on the docs: https://maniskill.readthedocs.io/en/dev/user_guide/getting_started/quickstart.html

## License

All rigid body environments in ManiSkill are licensed under fully permissive licenses (e.g., Apache-2.0).

The assets are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode).

## Citation

If you use ManiSkill2 or its assets and models, consider citing the following publication:

```
@inproceedings{gu2023maniskill2,
  title={ManiSkill2: A Unified Benchmark for Generalizable Manipulation Skills},
  author={Gu, Jiayuan and Xiang, Fanbo and Li, Xuanlin and Ling, Zhan and Liu, Xiqiaing and Mu, Tongzhou and Tang, Yihe and Tao, Stone and Wei, Xinyue and Yao, Yunchao and Yuan, Xiaodi and Xie, Pengwei and Huang, Zhiao and Chen, Rui and Su, Hao},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
