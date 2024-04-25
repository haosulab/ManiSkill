# {octicon}`package` Installation

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

:::{note}
While state-based simulation does not require any additional dependencies, a GPU with the Vulkan driver installed is required to enable rendering in ManiSkill. See [here](#vulkan) for how to install and configure Vulkan on Ubuntu.
:::

The rigid-body tasks, powered by SAPIEN, are ready to use after installation. Test your installation:

```bash
# Run an episode (at most 50 steps) of "PickCube-v1" (a rigid-body task) with random actions
# Or specify an task by "-e ${ENV_ID}"
python -m mani_skill.examples.demo_random_action
```

A docker image is also provided on [Docker Hub](https://hub.docker.com/repository/docker/haosulab/mani-skill/general) called  `haosulab/mani-skill` and its corresponding [Dockerfile](https://github.com/haosulab/ManiSkill/blob/main/docker/Dockerfile).

Once you are done here, you can head over to the [quickstart page](./quickstart.md) to try out some live demos and start to program with ManiSkill.
<!-- 
## Soft-body tasks / Warp (ManiSkill-version)

:::{note}
The following section is to install [NVIDIA Warp](https://github.com/NVIDIA/warp) for soft-body tasks. You can skip it if you do not need soft-body tasks yet.
:::

The soft-body tasks in ManiSkill are supported by SAPIEN and customized NVIDIA Warp. **CUDA toolkit >= 11.3 and gcc** are required. You can download and install the CUDA toolkit from the [offical website](https://developer.nvidia.com/cuda-downloads?target_os=Linux).

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

After CUDA is properly set up, compile Warp customized for ManiSkill:

``` bash
# If you encounter "ModuleNotFoundError: No module named 'warp'", please add warp_maniskill to the python path. 
export PYTHONPATH=/path/to/ManiSkill/warp_maniskill:$PYTHONPATH
# warp.so is generated under warp_maniskill/warp/bin
python -m warp_maniskill.build_lib
```

For soft-body tasks, you need to make sure only 1 CUDA device is visible:

``` bash
# Select the first CUDA device. Change 0 to other integer for other device.
export CUDA_VISIBLE_DEVICES=0
```

If multiple CUDA devices are visible, the task will give an error. If you
want to interactively visualize the task, you need to assign the id of
the GPU connected to your display (e.g., monitor screen).

:::{warning}
All soft-body tasks require runtime compilation and cache generation. The cache is generated in parallel. Thus, to avoid race conditions, before you create soft-body tasks in parallel, please make sure the cache is already generated. You can generate cache in advance by `python -m mani_skill.utils.precompile_mpm -e {ENV_ID}` (or without an option for all soft-body tasks).
::: -->

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
<!-- 
### Warp

If the soft-body task throws a **memory error**, you can try compiling Warp in the debug mode.

```bash
PYTHONPATH="$PWD"/warp_maniskill:$PYTHONPATH python -m warp_maniskill.build_lib --mode debug
```

Remember to compile again in the release mode after you finish debugging. In the debug mode, if the error becomes `unsupported toolchain`, it means you have a conflicting CUDA version. -->


If you still have some issues, you can check the NVIDIA drivers. First run

```bash
ldconfig -p | grep libGLX_nvidia
```

If `libGLX_nvidia.so` is not found, they it is likely that you have installed an incorrect driver. To get the right driver on linux, it is recommended to install `nvidia-driver-xxx` (do not use the ones with server in the package name) and to avoid using any other method of installation like a runfile

### Fast Kinematics

If you get an error showing

```bash
# ...
    import fast_kinematics
ImportError: libcudart.so.12: cannot open shared object file: No such file or directory
```

You are likely on CUDA 11 and need to use the CUDA 11 version of fast_kinematics, installable via 

```bash
pip install fast_kinematics==0.1.11
```

### Uninstallation

If `mani_skill` is installed through pip, run `pip uninstall mani-skill`.

:::{note}
There might exist some cache files (e.g., compiled shared library files, convex meshes generated by SAPIEN) generated in the package directory. To fully uninstall `mani_skill`, please remove those files manually.
:::
