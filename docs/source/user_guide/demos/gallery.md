# Demo Gallery

A page collecting all the videos that showcase various features of ManiSkill. The majority of these videos are generated via open-sourced code. The parts that are not yet open-sourced are labelled and are being cleaned up for release. 

## Parallel Rendering

<video preload="none" controls="True" width="100%" style="max-width: min(100%, 512px);" playsinline="true" poster="https://maniskill.ai/imgs/home/image-ky-01.webp"><source src="https://maniskill.ai/videos/feature-01.mp4" type="video/mp4"></video>
<caption>
    Parallel rendering of the AnymalC Quadruped robot controlled by a visual-based RL policy walking to a goal, showcasing a subset of the 1024 environments being rendered in parallel.
</caption>

## Heterogeneous Simulation

<video preload="none" controls="True" width="100%" style="max-width: min(100%, 512px);" playsinline="true" poster="https://maniskill.ai/imgs/home/image-ky-03.webp"><source src="https://maniskill.ai/videos/feature-03.mp4" type="video/mp4"></video>
<caption>
    Parallel heterogeneous simulation of the mobile manipulator Fetch robot opening different cabinets of different degrees of freedom, showcasing the ability to simulate different geometries and articulations in one GPU simulation. The robot is controlled by a state-based RL policy trained in 15 minutes on a single 4090 GPU.
</caption>

## Fast Visual Training Speed

<video preload="none" controls="True" width="49%" style="display: inline-block;" playsinline="true" poster="https://github.com/StoneT2000/paper-assets/raw/refs/heads/main/projects/mani_skill3/pickcube_low_closeup_thumb.jpg"><source src="https://github.com/StoneT2000/paper-assets/raw/refs/heads/main/projects/mani_skill3/pickcube_low_closeup.mp4" type="video/mp4"></video><video preload="none" controls="True" width="49%" style="display: inline-block;" playsinline="true" poster="https://github.com/StoneT2000/paper-assets/raw/refs/heads/main/projects/mani_skill3/pusht_low_close_thumb.jpg"><source src="https://github.com/StoneT2000/paper-assets/raw/refs/heads/main/projects/mani_skill3/pusht_low_close.mp4" type="video/mp4"></video>
<caption>
    Fast training speed of a state and vision-based RL policy for the PickCube and PushT tasks. With state inputs PickCube is solved in about 1 minute, PushT is solved in about 5 minutes. With visual inputs PickCube is solved in about 10 minutes and PushT is solved in about 50 minutes. PPO is used for training with 4096 parallel environments for state-based experiments and 1024 parallel environments for vision-based experiments, running on a single 4090 GPU.
</caption>


<!-- TODO find a place to host the larger videos instead of github -->
## Vision-Based Zero-shot Sim2Real Manipulation

We demonstrate some zero-shot sim2real manipulation results using the low-cost $300 Koch v1.1. robot arm and [🤗 LeRobot](https://github.com/huggingface/LeRobot) code for robot hardware interface/control. Policy is trained with PPO on RGB camera inputs and robot proprioceptive data for about an hour on a single 4090 GPU on a domain randomized simulation environment. This demo/code is on a public branch that has yet to be merged into the main branch.

### Real World Uncut Evaluation

<video preload="none" controls="True" width="100%" style="max-width: min(100%, 512px);" playsinline="true" poster="https://github.com/StoneT2000/paper-assets/raw/refs/heads/main/projects/mani_skill3/vision-based-sim2real/koch_arm_pickcube_eval_compressed_thumb.jpg"><source src="https://github.com/StoneT2000/paper-assets/raw/refs/heads/main/projects/mani_skill3/vision-based-sim2real/koch_arm_pickcube_eval_compressed.mp4" type="video/mp4"></video>
<caption>
    Real world evaluation of the PickCube task at 1x speed. 18/20 trials were successful where success is defined as the robot arm being able to pick up the cube and move it back to a rest position. In all 20/20 trials the robot arm was always able to grasp the cube. The camera observation fed to the policy is displayed on the phone screen.
</caption>
<br/>
<br/>
<p>Interestingly there are some untrained behaviors such as being able to pick up non cube-shaped objects, although we do not claim this kind of generalization always works.</p>

<video preload="none" controls="True" width="100%" style="max-width: min(100%, 512px);" playsinline="true" poster="https://github.com/StoneT2000/paper-assets/raw/refs/heads/main/projects/mani_skill3/vision-based-sim2real/koch_arm_pickcube_eval_ood_compressed_thumb.jpg"><source src="https://github.com/StoneT2000/paper-assets/raw/refs/heads/main/projects/mani_skill3/vision-based-sim2real/koch_arm_pickcube_eval_ood_compressed.mp4" type="video/mp4"></video>
<caption>
    Real world evaluation of the PickCube task at 1x speed on unseen object shapes.
</caption>

### Reset Distributions
<video preload="none" controls="True" width="32%" style="display: inline-block;" playsinline="true" poster="https://github.com/StoneT2000/paper-assets/raw/refs/heads/main/projects/mani_skill3/vision-based-sim2real/koch_arm_pickcube_reset_distribution_sim_thumb.jpg"><source src="https://github.com/StoneT2000/paper-assets/raw/refs/heads/main/projects/mani_skill3/vision-based-sim2real/koch_arm_pickcube_reset_distribution_sim.mp4" type="video/mp4"></video><video preload="none" controls="True" width="32%" style="display: inline-block;" playsinline="true" poster="https://github.com/StoneT2000/paper-assets/raw/refs/heads/main/projects/mani_skill3/vision-based-sim2real/koch_arm_pickcube_reset_distribution_sim_overlay_thumb.jpg"><source src="https://github.com/StoneT2000/paper-assets/raw/refs/heads/main/projects/mani_skill3/vision-based-sim2real/koch_arm_pickcube_reset_distribution_sim_overlay.mp4" type="video/mp4"> </video><video preload="none" controls="True" width="32%" style="display: inline-block;" playsinline="true" poster="https://github.com/StoneT2000/paper-assets/raw/refs/heads/main/projects/mani_skill3/vision-based-sim2real/koch_arm_pickcube_reset_distribution_real_thumb.jpg"><source src="https://github.com/StoneT2000/paper-assets/raw/refs/heads/main/projects/mani_skill3/vision-based-sim2real/koch_arm_pickcube_reset_distribution_real.mp4" type="video/mp4"></video>
<caption>
    Reset distribution of the PickCube task with the low-cost Koch v1.1. robot arm from LeRobot. Left: Simulation without overlay. Middle: Simulation with overlay. Right: Real world. Reset distribution here shows the domain randomizations all applied together to the simulation environment as well as the robustness testing we perform in the real world by testing on different cube sizes, colors, and poses.
</caption>


## Real2Sim Evaluation Environments 

<video preload="none" controls="True" width="49%" style="max-width: min(100%, 512px);display: inline-block;" playsinline="true" poster="https://github.com/StoneT2000/paper-assets/raw/refs/heads/main/projects/mani_skill3/real2sim/octo_base_put_eggplant_in_basket_thumb.jpg"><source src="https://github.com/StoneT2000/paper-assets/raw/refs/heads/main/projects/mani_skill3/real2sim/octo_base_put_eggplant_in_basket.mp4" type="video/mp4"></video>
<video preload="none" controls="True" width="49%" style="display: inline-block;" playsinline="true" poster="https://github.com/StoneT2000/paper-assets/raw/refs/heads/main/projects/mani_skill3/real2sim/octo_small_put_spoon_on_towel_thumb.jpg"><source src="https://github.com/StoneT2000/paper-assets/raw/refs/heads/main/projects/mani_skill3/real2sim/octo_small_put_spoon_on_towel.mp4" type="video/mp4"></video>
<video preload="none" controls="True" width="49%" style="display: inline-block;" playsinline="true" poster="https://github.com/StoneT2000/paper-assets/raw/refs/heads/main/projects/mani_skill3/real2sim/octo_small_stack_green_block_on_yellow_block_thumb.jpg"><source src="https://github.com/StoneT2000/paper-assets/raw/refs/heads/main/projects/mani_skill3/real2sim/octo_small_stack_green_block_on_yellow_block.mp4" type="video/mp4"></video>
<video preload="none" controls="True" width="49%" style="max-width: min(100%, 512px);display: inline-block;" playsinline="true" poster="https://github.com/StoneT2000/paper-assets/raw/refs/heads/main/projects/mani_skill3/real2sim/rt1x_put_carrot_on_plate_thumb.jpg"><source src="https://github.com/StoneT2000/paper-assets/raw/refs/heads/main/projects/mani_skill3/real2sim/rt1x_put_carrot_on_plate.mp4" type="video/mp4"></video>

We port over some of the Real2Sim evaluation environments from the SIMPLER project. The videos below show 4 different vision language action (VLA) models being evaluated on 4 different tasks (videos are originally from SIMPLER). These videos are subsets of the 128 environments that are being simulated and rendered in parallel to evaluate VLAs.


## Teleoperation

We provide a few teleoperation tools in ManiSkill. The most flexible of which is Virtual Reality (VR) based teleoperation. The teleoperation setup is being cleaned up at the moment and will be documented and open-sourced eventually.

<video preload="none" controls="True" width="100%" style="max-width: min(100%, 512px);" playsinline="true" poster="https://github.com/StoneT2000/paper-assets/raw/refs/heads/main/projects/mani_skill3/teleop/teleop_ability_hand_thumb.jpg"><source src="https://github.com/StoneT2000/paper-assets/raw/refs/heads/main/projects/mani_skill3/teleop/teleop_ability_hand_compressed.mp4" type="video/mp4"></video>
<caption>
    Teleoperation of a bi-manual 5-finger dextrous hand setup using the Meta Quest 3 system. The integrated VR teleoperation system enables 60 Hz streaming of 4K stereo video for low latency and smooth teleoperation.
</caption>
