# Diffusion Policy

Code for running the Diffusion Policy algorithm based on ["Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"](https://arxiv.org/abs/2303.04137v4). It is adapted from the [original code](https://github.com/real-stanford/diffusion_policy).

## Installation

To get started, we recommend using conda/mamba to create a new environment and install the dependencies

```bash
conda create -n diffusion-policy-ms python=3.9
conda activate diffusion-policy-ms
pip install -e .
```

## Demonstration Download and Preprocessing

By default for fast downloads and smaller file sizes, ManiSkill demonstrations are stored in a highly reduced/compressed format which includes not keeping any observation data. Run the command to download the demonstration and convert it to a format that includes observation data and the desired action space.

```bash
python -m mani_skill.utils.download_demo "PickCube-v1"
```

```bash
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o state \
  --save-traj --num-procs 10
```

Set -o to rgbd for RGBD observations. Note that the control mode can heavily influence how well Behavior Cloning performs. By default we recommend using `pd_joint_delta_pos` for control mode as all tasks can be solved with that control mode, although it is harder to learn with BC than `pd_ee_delta_pos` or `pd_ee_delta_pose` for robots that have those control modes. Finally, the type of demonstration data used can also impact performance, with typically neural network generated demonstrations being easier to learn from than human/motion planning generated demonstrations.

## Training

We provide scripts to train the diffusion policy on demonstrations. Note that some demonstrations are slow and can exceed the default max episode steps. In this case, you can use the `--max-episode-steps` flag to set a higher value. Most of the time 2x the default value is sufficient.


Example training, learning from 100 demonstrations generated via motionplanning in the PickCube-v1 task
```bash
python train.py --env-id PickCube-v1 \
  --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.cuda.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "cpu" --num-demos 100 --max_episode_steps 100 \
  --total_iters 30000 
```


## Train and Evaluate with GPU Simulation

You can also choose to train on trajectories generated in the GPU simulation and evaluate much faster with the GPU simulation. However as most demonstrations are usually generated in the CPU simulation (via motionplanning or teleoperation), you may observe worse performance when evaluating on the GPU simulation vs the CPU simulation. This can be partially alleviated by using the replay trajectory tool to try and replay trajectories back in the GPU simulation.

It is also recommended to not save videos if you are using a lot of parallel environments as the video size can get very large.

To replay trajectories in the GPU simulation, you can use the following command. Note that this can be a bit slow as the replay trajectory tool is currently not optimized for GPU parallelized environments.

```bash
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o state \
  --save-traj --num-procs 1 -b gpu --count 100 # process only 100 trajectories
```

Once our GPU backend demonstration dataset is ready, you can use the following command to train and evaluate on the GPU simulation.

```bash
python train.py --env-id PickCube-v1 \
  --demo-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.cuda.h5 \
  --control-mode "pd_ee_delta_pos" --sim-backend "gpu" --num-demos 100 --max_episode_steps 100 \
  --total_iters 30000 \
  --num-eval-envs 100 --no-capture-video
```

## Citation

If you use this baseline please cite the following
```
@inproceedings{DBLP:conf/rss/ChiFDXCBS23,
  author       = {Cheng Chi and
                  Siyuan Feng and
                  Yilun Du and
                  Zhenjia Xu and
                  Eric Cousineau and
                  Benjamin Burchfiel and
                  Shuran Song},
  editor       = {Kostas E. Bekris and
                  Kris Hauser and
                  Sylvia L. Herbert and
                  Jingjin Yu},
  title        = {Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  booktitle    = {Robotics: Science and Systems XIX, Daegu, Republic of Korea, July
                  10-14, 2023},
  year         = {2023},
  url          = {https://doi.org/10.15607/RSS.2023.XIX.026},
  doi          = {10.15607/RSS.2023.XIX.026},
  timestamp    = {Mon, 29 Apr 2024 21:28:50 +0200},
  biburl       = {https://dblp.org/rec/conf/rss/ChiFDXCBS23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```