# Installation

From pip (stable version):

```bash
# `mani-skill2` or `mani_skill2` is equivalent for pip
pip install mani-skill2
```

From github (latest commit):

```bash
pip install --upgrade git+https://github.com/haosulab/ManiSkill2.git
```

From source:

```bash
git clone https://github.com/haosulab/ManiSkill2.git
cd ManiSkill2 && pip install -e .
```

:::{note}
A GPU with the Vulkan driver installed is required to enable rendering in ManiSkill2. See [here](vulkan) for how to install and configure Vulkan on Ubuntu.
:::

The rigid-body environments, powered by SAPIEN, are ready to use after installation. Test your installation:

```bash
# Run an episode (at most 200 steps) of "PickCube-v0" (a rigid-body environment) with random actions
# Or specify an environment by "-e ${ENV_ID}"
python -m mani_skill2.examples.demo_random_action
```

Besides, we provide a docker image (`haosulab/mani-skill2`) on [Docker Hub](https://hub.docker.com/repository/docker/haosulab/mani-skill2/general), and its corresponding [Dockerfile](https://github.com/haosulab/ManiSkill2/blob/main/docker/Dockerfile).

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
# If you encounter "ModuleNotFoundError: No module named 'warp'", please add warp_maniskill to the python path. 
export PYTHONPATH=/path/to/ManiSkill2/warp_maniskill:$PYTHONPATH
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
All soft-body environments require runtime compilation and cache generation. The cache is generated in parallel. Thus, to avoid race conditions, before you create soft-body environments in parallel, please make sure the cache is already generated. You can generate cache in advance by `python -m mani_skill2.utils.precompile_mpm -e {ENV_ID}` (or without an option for all soft-body environments).
:::

## Troubleshooting

(vulkan)=

### Vulkan

To install Vulkan on Ubuntu:

```bash
sudo apt-get install libvulkan1
```

To test your installation of Vulkan:

```bash
sudo apt-get install vulkan-utils
vulkaninfo
```

If `vulkaninfo` fails to show the information about Vulkan, please check whether the following files exist:

- `/usr/share/vulkan/icd.d/nvidia_icd.json`
- `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`
- `/etc/vulkan/implicit_layer.d/nvidia_layers.json` (optional, but necessary for some GPUs like A100)

If `/usr/share/vulkan/icd.d/nvidia_icd.json` does not exist, try to create the file with the following content:

```json
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.2.155"
    }
}
```

If `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` does not exist, you can try `sudo apt-get install libglvnd-dev`. `10_nvidia.json` contains the following content:

```json
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
```

If `/etc/vulkan/implicit_layer.d/nvidia_layers.json` does not exist, try to create the file with the following content:

```json
{
    "file_format_version" : "1.0.0",
    "layer": {
        "name": "VK_LAYER_NV_optimus",
        "type": "INSTANCE",
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.2.155",
        "implementation_version" : "1",
        "description" : "NVIDIA Optimus layer",
        "functions": {
            "vkGetInstanceProcAddr": "vk_optimusGetInstanceProcAddr",
            "vkGetDeviceProcAddr": "vk_optimusGetDeviceProcAddr"
        },
        "enable_environment": {
            "__NV_PRIME_RENDER_OFFLOAD": "1"
        },
        "disable_environment": {
            "DISABLE_LAYER_NV_OPTIMUS_1": ""
        }
    }
}
```

More discussions can be found [here](https://github.com/haosulab/SAPIEN/issues/115).

---

The following errors can happen if the Vulkan driver is broken. Try to reinstall it following the above instructions.

- `RuntimeError: vk::Instance::enumeratePhysicalDevices: ErrorInitializationFailed`
- `Some required Vulkan extension is not present. You may not use the renderer to render, however, CPU resources will be still available.`
- `Segmentation fault (core dumped)`

### Warp

If the soft-body environment throws a **memory error**, you can try compiling Warp in the debug mode.

```bash
PYTHONPATH="$PWD"/warp_maniskill:$PYTHONPATH python -m warp_maniskill.build_lib --mode debug
```

Remember to compile again in the release mode after you finish debugging. In the debug mode, if the error becomes `unsupported toolchain`, it means you have a conflicting CUDA version.

### Uninstallation

If `mani_skill2` is installed through pip, run `pip uninstall mani-skill2`.

:::{note}
There might exist some cache files (e.g., compiled shared library files, convex meshes generated by SAPIEN) generated in the package directory. To fully uninstall `mani_skill2`, please remove those files manually.
:::
