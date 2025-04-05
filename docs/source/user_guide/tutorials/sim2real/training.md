# Part 2: Training with RL

For part 2, you should have already learned about making sim2real compatible environments and have modelled your real world task and robot/sensors already. If you skipped part 1 then you can just use our pre-built simulation environment designed for this tutorial which is a cube picking task with a Koch robot arm. Part 2 here will focus on training a vision-based policy with RL in simulation and then how to deploy it to the real world.


## 1 | Training in Simulation

We provide an example training script with Proximal Policy Optimization (PPO), which trains via RL a policy that picks up a cube from a table and lifts it up given just RGB inputs and proprioceptive based inputs. Understanding how PPO works is out of the scope of this tutorial, you can just our implementation to train the policy.


```
seed=0
python ppo_rgb.py -e KochPickCube-v1 \
    --num_envs=1024 --num-steps=16 --update_epochs=8 --num_minibatches=8 \
    --total_timesteps=100_000_000 \
    --exp-name="ppo-KochPickCube-v1-rgb-${seed}"
```


