# ManiSkill 3

# Project Installation Guide

This guide provides step-by-step instructions to set up the environment for this project, which is built upon ManiSkill3. The installation is tailored for a system with **CUDA 12.4** and **PyTorch 2.5.1**.

## Prerequisites

-   NVIDIA GPU with CUDA 12.4 compatible drivers.
-   [Conda](https://docs.conda.io/en/latest/miniconda.html) package manager installed.

## Installation Steps

### 1. Create and Activate Conda Environment

First, create a dedicated Conda environment for this project using Python 3.10. Then, activate the newly created environment.

```bash
# Create the environment named 'icra2025'
conda create -n icra2025 python=3.10

# Activate the environment
conda activate icra2025
```

### 2. Install PyTorch with CUDA 12.4

Install the correct version of PyTorch and its related libraries directly from the official `pytorch` and `nvidia` conda channels. This ensures that the GPU-accelerated libraries are linked correctly.

```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

### 3. Install Python Dependencies

Install all the required Python packages using the `requirements.txt` file. This file is configured to download packages compatible with our specific PyTorch and CUDA versions.

```bash
pip install -r requirements.txt
```

### 4. Install Custom ManiSkill Version

Finally, install the version of ManiSkill included in this repository in "editable" mode. The `-e` flag links the installation to this source directory, so any changes you make to the code are immediately effective in your environment.

```bash
pip install -e .
```

After this step, your environment is fully configured and ready to use.

---

## Verifying the Installation

To ensure PyTorch can correctly see and use your GPU, you can run the following Python code:

```python
import torch

if torch.cuda.is_available():
    print(f"✅ Success! PyTorch can see your GPU.")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("❌ Failure. PyTorch cannot see your GPU.")
```

---

## Citation

If you use ManiSkill3 (versions `mani_skill>=3.0.0`) in your work, please cite the ManiSkill3 paper:

```bibtex
@article{taomaniskill3,
  title={ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI},
  author={Stone Tao and Fanbo Xiang and Arth Shukla and Yuzhe Qin and Xander Hinrichsen and Xiaodi Yuan and Chen Bao and Xinsong Lin and Yulin Liu and Tse-kai Chan and Yuan Gao and Xuanlin Li and Tongzhou Mu and Nan Xiao and Arnav Gurha and Viswesh Nagaswamy Rajesh and Yong Woo Choi and Yen-Ru Chen and Zhiao Huang and Roberto Calandra and Rui Chen and Shan Luo and Hao Su},
  journal = {Robotics: Science and Systems},
  year={2025},
}
```

## License

All rigid body environments in ManiSkill are licensed under fully permissive licenses (e.g., Apache-2.0).

The assets are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode).
