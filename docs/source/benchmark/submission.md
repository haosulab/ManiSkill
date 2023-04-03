# Submission

To participate in the ManiSkill2 challenge, please register on the [challenge website](https://sapien.ucsd.edu/challenges/maniskill/). After registering an account, [create/join a team](https://sapien.ucsd.edu/challenges/maniskill/challenges/ms2/team). After creating/joining a team, you will be allowed to create submissions.

To submit to the challenge, you need to submit a URL to your docker image which contains your codes and dependencies (e.g., model weights). Before submitting, you should test the submission docker locally. Instructions for local evaluation and online submission are provided below.

In brief, you need to:

- Create a file named "user_solution.py", and implement a `UserPolicy` in it. An example is provided [here](https://github.com/haosulab/ManiSkill2/tree/main/examples/submission).
- Build a docker image that includes "user_solution.py" and other codes and dependencies (e.g., model weights).
- Test the docker image locally, give it a unique tag, and push it to a public docker registry.

Please see the [evaluation script](https://github.com/haosulab/ManiSkill2/tree/main/mani_skill2/evaluation/run_evaluation.py) for how your submission will be evaluated. You can locally test your submission by `python -m mani_skill2.evaluation.run_evaluation ...`. We will use the same script (with a different subclass of `BaseEvaluator` and held-out configuration files) to evaluate your online submission.

## Create and locally verify a solution

First, you need to create a file named `user_solution.py`, where a `UserPolicy` is implemented. The evaluation script attempts to load the solution through `from user_solution import UserPolicy`. Therefore, you need to ensure `user_solution.py` can be imported. The following commands show how to evaluate your solution locally.

```bash
# Add your submission to PYTHONPATH. Ensure that "user_solution.py" can be found to import.
# Assume that "user_solution.py" is under ${PATH_TO_YOUR_CODES_IN_HOST}
export PYTHONPATH=${PATH_TO_YOUR_CODES_IN_HOST}:$PYTHONPATH

# Test whether the user solution can be imported
# python -c "from user_solution import UserPolicy"

# Run evaluation. The result will be saved to ${OUTPUT_DIR}.
ENV_ID="PickCube-v0" OUTPUT_DIR="tmp" NUM_EPISODES=1
python -m mani_skill2.evaluation.run_evaluation -e ${ENV_ID} -o ${OUTPUT_DIR} -n ${NUM_EPISODES}
```

## Build a docker image

Install [nvidia-docker v2](https://github.com/NVIDIA/nvidia-docker) following instructions here: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker>. Note: only supports Linux; no Windows or MacOS.

We provide a base image at Dockerhub: `haosulab/mani-skill2:latest`. It is based on `nvidia/cudagl:11.3.1-devel-ubuntu20.04`. The corresponding Dockerfile is [here](https://github.com/haosulab/ManiSkill2/blob/main/docker/Dockerfile).

Here is an example of how to customize the base docker image.

```Dockerfile
FROM haosulab/mani-skill2:latest

# Install additional python packages you need
RUN pip install torch==1.12.1

# Copy your codes (including user_solution.py) and model weights
COPY ${YOUR_CODES_AND_WEIGHTS_IN_HOST} ${YOUR_CODES_AND_WEIGHTS_IN_CONTAINER}
ENV PYTHONPATH ${YOUR_CODES_AND_WEIGHTS_IN_CONTAINER}:$PYTHONPATH
```

Given a Dockerfile, you can build your submission docker image.

```bash
# It is suggested to run this command under the directory containing Dockerfile
# See https://docs.docker.com/engine/reference/commandline/build/ for more details
# Here PATH_TO_BUILD_CONTEXT is the local path context under which the docker building instructions like COPY should reference the files.
docker build -f ${PATH_TO_YOUR_DOCKERFILE} ${PATH_TO_BUILD_CONTEXT} -t mani-skill2-submission
```

Finally, you can tag your image and push it to a public docker registry (e.g., [Dockerhub](https://hub.docker.com/)).

```bash
# Tag your local image before uploading it to Dockerhub
docker tag mani-skill2-submission ${DOCKERHUB_USER_NAME}/mani-skill2-submission:${UNIQUE_TAG}
docker push ${DOCKERHUB_USER_NAME}/mani-skill2-submission:${UNIQUE_TAG}
```

As Dockerhub's registry is public, we recommend you create a new anonymous docker account to prevent people from finding it.

:::{warning}
We only accept the docker image the size of which is smaller than 24GB. Please refer to <https://docs.docker.com/develop/develop-images/dockerfile_best-practices/> to reduce size.
:::

## Test the docker image locally

Run the following script to test your docker image locally. If this works, it is ready for submission to the challenge.

```bash
export DOCKER_IMAGE=mani-skill2-submission:${UNIQUE_TAG}
export CONTAINER_NAME=mani-skill2-evaluation
# Initialize a detached container. If you are evaluating tasks with extra assets, you need to mount the directory containing downloaded assets to the container.
docker run -d --rm --gpus all --name ${CONTAINER_NAME} \
    -v ${PATH_TO_MS2_ASSET_DIR}:/data \
    ${DOCKER_IMAGE}
# Interactive debug
docker exec -it ${CONTAINER_NAME} /bin/bash
# Or run evaluation
docker exec -it ${CONTAINER_NAME} /bin/bash -c "export MS2_ASSET_DIR=/data; python -m mani_skill2.evaluation.run_evaluation -e PickCube-v0 -o /eval_results/PickCube-v0 -n 1"
# Finally, you can delete the container
docker kill ${CONTAINER_NAME}
```

## Online Submission

Once you have built and pushed a docker image, you are ready to submit to the competition. Go to the competition [submissions page](https://sapien.ucsd.edu/challenges/maniskill/challenges/ms2/submit) and give your submission a name and enter the docker image name+tag (format: `registry.hub.docker.com/USERNAME/IMG_NAME:TAG`; Do not use the `latest` tag). Then select which track you are submitting to. Lastly, tick/untick which tasks you would like to evaluate your submission on.

To ensure reproducibility, we do not allow you to submit the same docker image and tag twice, we require you to give a new tag to your image before submitting. You can create a new tag as so `docker tag <image_name> <image_name>:<tag_name>`

We **strongly recommend** you only tick the tasks you want to evaluate as we rate limit team submissions by the number of tasks evaluated each day. Each team is given a budget of 50 task evaluations each day. Once the number of task evaluations in a day has gone over 50, we disable submissions for your team. This budget resets every day at 8:00 AM UTC.
