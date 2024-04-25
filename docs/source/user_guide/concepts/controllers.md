# Controllers / Action Spaces

Controllers are interfaces between users/policies and robots. Whenever you take a step in an environment and provide an action, that action is sent to a chosen controller that converts actions to control signals to the robot. At the lowest level, all robots in simulation are controlled via joint position or joint velocity controls, effectively specifying where / how fast each joint should go.

For example, the `arm_pd_ee_delta_pose` controller takes the relative movement of the end-effector as input, and uses [inverse kinematics](https://en.wikipedia.org/wiki/Inverse_kinematics) to convert input actions to target positions of robot joints. The robot uses a [PD controller](https://en.wikipedia.org/wiki/PID_controller) to drive motors to achieve target joint positions.

There are a few key elements to remember about controllers in ManiSkill
- The controller defines the action space of a task 
- The robot can have separate controllers for different groups of joints. The action space is a concatenation of the action spaces of all controllers
- A single robot may have several sets of controllers that can be used


<!-- Note that while `pd_ee_delta_pose` type controllers that use IK may be more sample efficient to train / learn from for RL workflows, in GPU simulation running these controllers is not that fast and may slow down RL training. -->

The next section will detail each of the pre-built controllers and what they do

## Prebuilt Controllers (Docs WIP)


### Passive

```python
from mani_skill.agents.controllers import PassiveControllerConfig
```

This controller lets you enforce given joints to be not controlled by actions. An example of this is used for the [CartPole environment](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/envs/tasks/control/cartpole.py) which defines the CartPole robot as having passive control over the hinge joint of the CartPole (the CartPole task only allows control of the sliding box).

### PD Joint Position

```python
from mani_skill.agents.controllers import PDJointPosControllerConfig
```

With a PD controller, controls the joint positions of the given joints via actions.

## Deep Dive Example of the Franka Emika Panda Robot Controllers:

To help detail how controllers work in detail, below we explain with formulae how the controllers are controlling the root in simulation.

### Terminology

- fixed joint: a joint that can not be controlled. The degree of freedom (DoF) is 0.
- `qpos` ( $q$ ): controllable joint positions
- `qvel` ( $\dot{q}$ ): controllable joint velocities
- target joint position ( $\bar{q}$ ): target position of the motor which drives the joint
- target joint velocity ( $\bar{\dot{q}}$ ): target velocity of the motor which drives the joint
- PD controller: control loop based on $F(t) = K_p (\bar{q}(t) - q(t)) + K_d (\bar{\dot{q}}(t) - \dot{q}(t))$. $K_p$ (stiffness) and $K_d$ (damping) are hyperparameters. $F(t)$ is the force of the motor.
- Augmented PD controller: Passive forces (like gravity) can be compensated by disabling gravity (they can be computed but this is slow and not necessary).
- action ( $a$ ): the input of the controller, also the output of the policy

### Supported Controllers

The robot is [Franka Emika](https://github.com/frankaemika/franka_ros), a.k.a Panda. The DoF of the arm is 7. **We use the tool center point (TCP), which is the center between two fingers, as the end-effector.**

All "pd" controllers translate input actions to target joint positions $\bar{q}$ (and velocities) for the internal built-in PD controller. **All the controllers have a normalized action space ([-1, 1]), except `arm_pd_joint_pos` and `arm_pd_joint_pos_vel`.**

For simplicity, we use the name of the arm controller to represent each combination of the arm and gripper controllers, since there is only one gripper controller currently. For example, `pd_joint_delta_pos` is short for `arm_pd_joint_delta_pos` + `gripper_pd_joint_pos`.

### Arm controllers

- arm_pd_joint_pos (7-dim): The input is $\bar{q}$. It can be used for motion planning, but note that the target velocity is always 0.
- arm_pd_joint_delta_pos (7-dim):  $\bar{q}(t)=q(t) + a$
- arm_pd_joint_target_delta_pos (7-dim): $\bar{q}(t)=\bar{q}(t-1) + a$
- arm_pd_ee_delta_pos (3-dim): only control position ( $p$ ) without rotation ( $R$ ).
  
  $\bar{p}(t)=p(t)+a$, $\bar{R}(t)=R(t)$, $\bar{q}(t)=IK(\bar{p}(t), \bar{R}(t))$

- arm_pd_ee_delta_pose (6-dim): both position ( $p$ ) and rotation ( $R$ ) are controlled. Rotation is represented as axis-angle in the end-effector frame.

  $\bar{p}(t)=p(t)+a_{p}$, $\bar{R}(t)=R(t) \cdot e^{[a_{R}]_{\times}}$, $\bar{q}(t)=IK(\bar{p}(t), \bar{R}(t))$

- arm_pd_ee_target_delta_pos (3-dim): $\bar{p}(t)=\bar{p}(t-1)+a$, $\bar{R}(t)=R(t)$
- arm_pd_ee_target_delta_pose (6-dim): $\bar{T}(t)= T_{a} \cdot \bar{T}(t-1)$. $T$ is the transformation of the end-effector. $T_a$ is the delta pose induced by the action.
- arm_pd_joint_vel (7-dim): only control target joint velocities. Note that the stiffness $K_p$ is set to 0.
- arm_pd_joint_pos_vel (14-dim): the extension of `arm_pd_joint_pos` to support target velocities
- arm_pd_joint_delta_pos_vel (14-dim): the extension of `arm_pd_joint_delta_pos` to support target velocities

### Gripper controllers (optional)

- gripper_pd_joint_pos (1-dim): Note that we force two gripper fingers to have the same target position. Thus, it is like a "mimic" joint.

<!-- ## Mobile Manipulator

The mobile manipulator is a combination of sciurus17 connector and one or two Panda arms. The controller is named `base_{}_arm_{}`. Except for the base controller, the arm and gripper controllers are the same as in the stationary manipulator.

### Base controllers

- base_pd_joint_vel (4-dim): only control target velocities. The first 2 dimensions stand for egocentric xy-plane linear velocity and the 3rd dimension stands for egocentric z-axis angular velocity. The 4th dimension stands for velocity to adjust torso. -->
