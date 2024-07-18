# Action Chunking Transformer (ACT)

Code for running the PPO RL algorithm is adapted from [the original ACT repository](https://github.com/tonyzhaozh/act) and the original paper ["Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"](https://arxiv.org/abs/2304.13705)

First clone the code:

```bash
git clone https://github.com/tonyzhaozh/act
```

To install following the ACT repo's install instructions run

```bash
# conda create -n aloha python=3.8.10
conda create -n aloha python=3.9
conda activate aloha
pip install torchvision
pip install torch
pip install pyquaternion
pip install pyyaml
pip install rospkg
pip install pexpect
# pip install mujoco==2.3.7
# pip install dm_control==1.0.14
pip install opencv-python
pip install matplotlib
pip install einops
pip install packaging
pip install h5py
pip install ipython
pip install wandb
cd act/detr && pip install -e .
```

Then install ManiSkill

```bash
pip install mani_skill
```


## Download Data


```bash
python3 record_sim_episodes.py \
  --task_name sim_transfer_cube_scripted \
  --dataset_dir demos \
  --num_episodes 50

python3 imitate_episodes.py \
  --task_name sim_transfer_cube_scripted \
  --ckpt_dir ckpts \
  --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
  --num_epochs 2000  --lr 1e-5 \
  --seed 0

python3 imitate_episodes.py \
  --task_name sim_transfer_cube_scripted \
  --ckpt_dir ckpts \
  --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
  --num_epochs 2000  --lr 1e-5 \
  --seed 0 --eval
```

## Train

```bash
python3 imitate_episodes.py --task_name sim_transfer_cube_scripted --ckpt_dir ckpts --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --seed 0
```
