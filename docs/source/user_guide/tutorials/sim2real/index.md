# Sim2Real Manipulation

We provide a two-part tutorial on sim2real for simple manipulation tasks. By the end of this tutorial you will learn how to build a simulation environment for training, convert it with minimal code to a real environment interface, and train a vision-based policy in simulation that can tackle real tasks from just RGB input and joint position signals. Note that this tutorial does not solve the sim2real problem, but is one approach that we have optimized and refined for accessibility and reproducibility.

```{toctree}
:titlesonly:

setup
training
```

At a high-level the approach of this tutorial is to create heavily domain randomized simulation environments that approximate the real world setting, then using reinforcement learning to train vision-based policies with fast GPU simulation+rendering in ManiSkill. This particular tutorial is mostly agnostic to the robot hardware you use but for the purposes of accessibility we will use the [🤗 LeRobot](https://github.com/huggingface/lerobot) system along with the low-cost [Koch robot arm](https://github.com/jess-moss/koch-v1-1). By the end you can deploy your robot to do zero-shot tackle tasks in the real-world! The video below shows the result, a RGB based policy trained for about 1 hour on a 4090 GPU in simulation being deployed to the real-world to pick up objects of varying sizes/colors.

<video preload="none" controls="True" width="100%" style="max-width: min(100%, 512px);" playsinline="true" poster="https://github.com/StoneT2000/paper-assets/raw/refs/heads/main/projects/mani_skill3/vision-based-sim2real/koch_arm_pickcube_eval_compressed_thumb.jpg"><source src="https://github.com/StoneT2000/paper-assets/raw/refs/heads/main/projects/mani_skill3/vision-based-sim2real/koch_arm_pickcube_eval_compressed.mp4" type="video/mp4"></video>


There is a lot of room to improve this simple sim2real approach (e.g. using a few real-world demonstrations) but we release this system and tutorial as a simple starting point for accessible sim2real manipulation research.


<!-- 
For sim2real one typically needs to align dynamics and visual data, a (still) very difficult problem. ManiSkill provides a few utilities to help minimize the amount of extra code you need to write and streamline the process of deploying policies trained in ManiSkill simulation to the real world. The approach and tools provided here in no way solve the sim2real problem, but are a step towards making it more accessible to work on sim2real transfer and address some of the problems. A recommended pre-requisite to making sim2real environments in this tutorial is to first learn how to create simulation tasks in the [custom tasks tutorial](./custom_tasks/intro.md).

We first describe at a high-level some of the features of the {py:class}`mani_skill.envs.sim2real_env.Sim2RealEnv` class that we provide that helps streamline the process of creating sim2real environments. Then the full tutorial will give examples and step-by-step instructions on how to make your own Sim2Real environments using the highly accessible / low-cost [LeRobot](https://github.com/huggingface/lerobot) system for easy robot/sensor setups. Coming soon will also include simple RGB-based sim2real deployment of policies trained with RL entirely in simulation (a demo showcase of that is [in the demo gallery](../demos/gallery.md#vision-based-zero-shot-sim2real-manipulation) if you are interested). -->

