# Docker

Docker provides a convenient way to package software into standardized units for development, shipment and deployment. See the [official website](https://www.docker.com/resources/what-container/) for more details about Docker. [NVIDIA Container Tookit](https://github.com/NVIDIA/nvidia-docker) enables users to build and run GPU accelerated Docker containers.

First, install [nvidia-docker v2](https://github.com/NVIDIA/nvidia-docker) following the [official instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker). It is recommended to complete post-install steps for Linux.

To verify the installation:

```bash
# You should be able to run this without sudo.
docker run hello-world
```

## Run ManiSkill2 in Docker

We provide a docker image (`haosulab/mani-skill2`) and its corresponding [Dockerfile](https://github.com/haosulab/ManiSkill2/blob/main/docker/Dockerfile).

```bash
docker pull haosulab/mani-skill2
docker run --rm -it --gpus all haosulab/mani-skill2 python -m mani_skill2.examples.demo_random_action
```

## Run GUI Applications

To run GUI applications from the docker container (the host is attached with a display), you need to add extra options to the `docker run` command:

```bash
# Allow local X11 connections
xhost +local:root
# Run ManiSkill2 docker image with the NVIDIA GPU
docker run --rm -it --gpus all \
    -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY \
    haosulab/mani-skill2 \
    python -m mani_skill2.examples.demo_manual_control -e PickCube-v0 --enable-sapien-viewer
```

To run GUI applications on a headless server, we present a solution based on `x11vnc` and `fluxbox`.

```bash
# https://www.richud.com/wiki/Ubuntu_Fluxbox_GUI_with_x11vnc_and_Xvfb
docker run --rm --gpus all -p 5900:5900 \
    haosulab/mani-skill2 \
    apt update && bash -c "apt install -yqq x11vnc fluxbox && x11vnc -create -env FD_PROG=/usr/bin/fluxbox  -env X11VNC_FINDDISPLAY_ALWAYS_FAILS=1 -env X11VNC_CREATE_GEOM=${1:-1920x1080x16} -gone 'pkill Xvfb' -nopw"
```

Then, forward the port of VNC (5900 by default) to the local host. On your local machine, install a [VNC viewer](https://www.realvnc.com/en/connect/download/viewer/) and connect to the localhost port(e.g. localhost:5900).

---

**References**:

- <https://medium.com/@benjamin.botto/opengl-and-cuda-applications-in-docker-af0eece000f1>
- <http://wiki.ros.org/docker/Tutorials/GUI>
- <https://github.com/haosulab/ManiSkill2/issues/62>
