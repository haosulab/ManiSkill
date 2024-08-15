# Stable Baselines 3

The example.py code shows a very simple example of how to use Stable Baselines 3 with ManiSkill 3 via a simple wrapper. These are not tuned as much compared to the [recommended PPO baseline code](https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/ppo) so we cannot guarantee good performance from using Stable Baselines 3. Moreover, currently Stable Baselines 3 is not optimized for GPU vectorized environments, so it will train a bit slower.

If you use Stable Baselines 3 please cite the following 

```
@article{stable-baselines3,
  author  = {Antonin Raffin and Ashley Hill and Adam Gleave and Anssi Kanervisto and Maximilian Ernestus and Noah Dormann},
  title   = {Stable-Baselines3: Reliable Reinforcement Learning Implementations},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {268},
  pages   = {1-8},
  url     = {http://jmlr.org/papers/v22/20-1364.html}
}
```