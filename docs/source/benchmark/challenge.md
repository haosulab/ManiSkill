# Challenge

To participate in the ManiSkill2022 challenge, please register on the [challenge website](https://sapien.ucsd.edu/challenges/maniskill/2022/). After registering an account, [create/join a team](https://sapien.ucsd.edu/challenges/maniskill/challenges/ms2022/team). After creating/joining a team, you will be allowed to create submissions.

To submit to the challenge, you need to submit a URL to your docker image which contains your code as well as model weights. Before submitting however, you should test the submission docker locally. Instructions for local evaluation and online submission are provided below.

In brief, you need to:

- Create a file named "user_solution.py", and implement a `UserPolicy` in it. An example is provided [here](https://github.com/haosulab/ManiSkill2022/tree/main/examples/submission)
- Build a docker image, and copy "user_solution.py" and other codes as well as model weights into it.
- Test the docker image locally, give it a unique tag, and push it to a public docker registry.

The evaluation script is *mani_skill2/evaluation/run_evaluation.py*. It shows how your submission will be evaluated, and you can run it locally to test whether your submission is valid. We use the same script to evaluate your online submission.

## Create and locally verify a solution

First, create a file named `user_solution.py`, and implement a `UserPolicy` in it. Let the directory containing `user_solution.py` be `PATH_TO_YOUR_CODES_IN_HOST`.

We have provided an evaluation script in `mani_skill2/evaluation/run_evaluation.py` to locally verify `user_solution.py`. This evaluation script attempts to load the solution through `from user_solution import UserPolicy`. Therefore, to test evaluation, you need to add `PATH_TO_YOUR_CODES_IN_HOST` to `PYTHONPATH`, and ensure that "user_solution.py" can be found to import. After this, you can use our evaluation script to test your policy. Use the following commands to conduct these steps:

```bash
# Add your submission to PYTHONPATH. Ensure that "user_solution.py" can be found to import.
export PYTHONPATH=${PATH_TO_YOUR_CODES_IN_HOST}:$PYTHONPATH

# Test whether the user solution can be imported
# python -c "from user_solution import UserPolicy"

# Run evaluation. The result will be saved to ${OUTPUT_DIR}.
ENV_ID="PickCube-v0" OUTPUT_DIR="tmp"
python -m mani_skill2.evaluation.run_evaluation -e ${ENV_ID} -o ${OUTPUT_DIR}
```

## Build a docker image

Install [nvidia-docker v2](https://github.com/NVIDIA/nvidia-docker) following instructions here: <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker>. Note: only supports Linux; no Windows or MacOS.

We provide a base image at Dockerhub: `haosulab/maniskill2022-challenge:latest`. It is based on `nvidia/cudagl:11.3.1-devel-ubuntu20.04`. The corresponding Dockerfile is at [docker/Dockerfile](https://github.com/haosulab/ManiSkill2/blob/main/docker/Dockerfile).

Here is an example of how to customize the base docker image.

```Dockerfile
FROM haosulab/maniskill2022-challenge:latest

# Install additional python packages you need
RUN conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch && pip install pytransform3d

# Copy your codes (including user_solution.py) and model weights
COPY ${YOUR_CODES_AND_WEIGHTS_IN_HOST} ${YOUR_CODES_AND_WEIGHTS_IN_CONTAINER}
ENV PYTHONPATH ${YOUR_CODES_AND_WEIGHTS_IN_CONTAINER}:$PYTHONPATH
```
Note that we use PyTorch 1.11.0. This is because 1.12.0 has an [issue](https://github.com/pytorch/pytorch/issues/80809).

With the Dockerfile ready, you can build your submission docker image.

```bash
# It is suggested to run this command under the directory containing Dockerfile
docker build -f ${PATH_TO_YOUR_DOCKERFILE} -t maniskill2022-submission
```

Finally, you can tag your image and push to a public docker registry (e.g., [Dockerhub](https://hub.docker.com/)).

```bash
# Tag your local image before uploading it to Dockerhub
docker tag maniskill2022-submission ${DOCKERHUB_USER_NAME}/maniskill2022-submission:test_1
docker push ${DOCKERHUB_USER_NAME}/maniskill2022-submission
```

As Dockerhub's registry is public, we recommend you to create a new anonymous docker account to prevent people from finding it.

We only accept the docker image the size of which is smaller than 24GB. Please refer to <https://docs.docker.com/develop/develop-images/dockerfile_best-practices/> to reduce size.

## Test the docker image locally

Run the following script to test your docker image locally. If this works, it is ready for submission to the challenge.

```bash
DOCKER_IMAGE=maniskill2022-submission
docker run -d --rm --gpus all --name maniskill2022-evaluation \
    -v ${ABSOLUTE_PATH_TO_ManiSkill2}:/root/ManiSkill2 \
    ${DOCKER_IMAGE}
# Interactive debug
docker exec -it maniskill2022-evaluation /bin/bash
# Or run evaluation
docker exec -it maniskill2022-evaluation /bin/bash -c "export PYTHONPATH=/root/ManiSkill2:$PYTHONPATH; python -m mani_skill2.evaluation.run_evaluation -e PickCube-v0 -o /root/ManiSkill2/eval_results/PickCube-v0"
# Finally, you can delete the container
docker kill maniskill2022-evaluation
```

## Online Submission

Once you have built and pushed a docker image, you are ready to submit to the competition. Go to the competition [submissions page](https://sapien.ucsd.edu/challenges/maniskill/challenges/ms2022/submit) and give your submission a name and enter the docker image name+tag (format: `registry.hub.docker.com/USERNAME/IMG_NAME:TAG`; Do not use the `latest` tag). Then select which track you are submitting to. Lastly, tick/untick which tasks you would like to evaluate your submission on.

To ensure reproducibility, we do not allow you to submit the same docker image and tag twice, we require you to give a new tag to your image before submitting. You can create a new tag as so `docker tag <image_name> <image_name>:<tag_name>`

We **strongly recommend** you to only tick the tasks you want to evaluate on as we rate limit team submissions by the number of tasks evaluated each day. Each team is given a budget of 50 task evaluations each day. Once the number of task evaluations in a day has gone over 50, we disable submissions for your team. This budget resets every day at 8:00 AM UTC time.
