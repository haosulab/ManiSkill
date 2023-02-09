# Installation

From pip:

```bash
pip install mani_skill2
```

From source:

```bash
git clone https://github.com/haosulab/ManiSkill2.git
cd ManiSkill && pip install -e .
```

Test your installation:

```bash
python -m mani_skill2.examples.demo_random_action
```

## Warp (ManiSkill2-version)

:::{note}
The following section is to install [NVIDIA Warp](https://github.com/NVIDIA/warp) for soft-body environments. You can skip it if you do not need soft-body environments yet.
:::

The soft-body environments in ManiSkill2 are supported by SAPIEN and customized NVIDIA Warp. **CUDA toolkit >= 11.3 and gcc** are required. You can download and install the CUDA toolkit from the [offical website](https://developer.nvidia.com/cuda-downloads?target_os=Linux).

Assuming the CUDA toolkit is installed at `/usr/local/cuda`, you need to ensure `CUDA_PATH` or `CUDA_HOME` is set properly:

```bash
export CUDA_PATH=/usr/local/cuda

# The following command should print a CUDA compiler version >= 11.3
${CUDA_PATH}/bin/nvcc --version

# The following command should output a valid gcc version
gcc --version
```

:::{note}
If `nvcc` is included in `$PATH`, we will try to figure out the variable `CUDA_PATH` automatically.
:::

After CUDA is properly set up, compile Warp customized for ManiSkill2:

``` bash
# warp.so is generated under warp_maniskill/warp/bin
python -m warp_maniskill.build_lib
```

For soft-body environments, you need to make sure only 1 CUDA device is visible:

``` bash
# Select the first CUDA device. Change 0 to other integer for other device.
export CUDA_VISIBLE_DEVICES=0
```

If multiple CUDA devices are visible, the environment will give an error. If you
want to interactively visualize the environment, you need to assign the id of
the GPU connected to your display (e.g., monitor screen).

:::{warning}
All soft-body environments require runtime compilation and cache generation. Cache is generated in parallel. Thus, to avoid race condition, before you create soft-body environments in parallel, please make sure cache is already generated. You can generate cache in advance by `python -m mani_skill2.utils.precompile_mpm -e {ENV_ID}` (or without option for all soft-body environments).
:::

## Troubleshooting

If the soft-body environment throws a **memory error**, you can try compiling Warp in the debug mode.

```bash
python -m warp_maniskill.build_lib --mode debug
```

Remember to compile again in the release mode after you finish debugging.

In the debug mode, if the error becomes `unsupported toolchain`, it means you have a conflicting CUDA version.
