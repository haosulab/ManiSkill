# Custom Robots

ManiSkill supports importing robots and assets via URDF and MJCF definitions. As ManiSkill was designed to allow one to flexibly change mounted sensors, controllers, and robot embodiments, ManiSkill at the moment does not automatically import those elements into the created robot, they must be defined by you.

In summary, the following elements must be completed before a robot is usable.

1. Create a robot class and specify a uid, as well as a urdf/mjcf file to import
2. Define robot controller(s) (e.g. PD joint position control)
3. (Optional): Define mounted sensors / sensors relative to the robot (e.g. wrist cameras)

This tutorial will guide you through on how to implement the Panda robot in ManiSkill. It will also cover some tips/tricks for modelling other categories of robots to ensure the simulation runs fast and accurately (e.g., mobile manipulators like Fetch, quadrupeds like Anymal)


## 1. Robot Class and Importing Robot

To create your own robot (also known as an Agent) you need to inherit the `BaseAgent` class, give it name, and optionally register the agent.

```python
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.registration import register_agent
@register_agent()
class MyPanda(BaseAgent):
    uid = "my_panda"
```

Registering the agent allows you to create environments that instantiate your robot for you via a string uid. Generally you can now run the following to visualize the robot by ID. 

```python
env = gym.make("EmptyEnv-v1", robot_uids="my_panda")
```

With the base robot class started, now we need to import a robot definition from a URDF or MJCF

### Importing the Robot

To import a URDF/MJCF file, you simply provide a path to the definition file and ManiSkill handles the importing. ManiSkill will automatically create the appropriate articulation with links and joints defined in the given definition file. Internally ManiSkill uses either the URDFLoader or MJCFLoader class in the `mani_skill.utils.building` module.

#### URDF

To get started, you first need to get a valid URDF file like this one for the Panda robot: https://github.com/haosulab/ManiSkill2/blob/dev/mani_skill/assets/robots/panda/panda_v2.urdf

Then in the agent class add

```python
class MyPanda(BaseAgent):
    uid = "my_panda"
    urdf_path = f"path/to/your/robot.urdf"
```



<!-- For a starting template check out https://github.com/haosulab/ManiSkill2/blob/dev/mani_skill/agents/robots/_template/template_robot.py -->

#### Mujoco MJCF

ManiSkill supports importing [Mujoco's MJCF format](https://mujoco.readthedocs.io/en/latest/modeling.html) of files to load robots (and other objects), although not all features are supported.

For example code that loads the robot and the scene see https://github.com/haosulab/ManiSkill2/blob/dev/mani_skill/envs/tasks/control/cartpole.py


At the moment, the following are not supported:
- Procedural texture generation
- Importing motors and solver configurations
- Correct use of contype and conaffinity attributes (contype of 0 means no collision mesh is added, otherwise it is always added)
- Correct use of groups (at the moment anything in group 0 and 2 can be seen, other groups are hidden all the time)

These may be supported in the future so stay tuned for updates.

## 2. Defining Controllers

ManiSkill permits defining multiple controllers for a single agent/robot, allowing for easy research and testing on different controllers to explore problems like sim2real and more. 


In general, for each **active joint** in a robot (which is represented as an Articulation object), you must define some controller over it. You can check which joints are active by checking your .urdf file and looking for all joints where the type attribute is not `"fixed"`. Another simple and easy way to check is to directly use the URDF or MJCF loader yourself and print the active joints which can be done as so

```python
import sapien
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.building import URDFLoader
loader = URDFLoader()
loader.set_scene(ManiSkillScene())
robot = loader.load("path/to/robot.urdf")
print(robot.active_joints_map.keys())
# For Panda it prints
# dict_keys(['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_finger_joint1', 'panda_finger_joint2'])
```

By default, ManiSkill provides out of the box several kinds of controllers, the main ones that you may typically use are the PDJointPosController, PDEEPoseController, and the PassiveController.

You can find more information on provided controllers and how some work in detail on the [controllers documentation](../concepts/controllers)

In brief, we will show how to work with both the PDJointPosController and PDJointPosMimicController here as they are both defined for the Panda robot. The first is for controlling the arm, and the mimic controller is for controlling the grippers.

To define controllers, you need to implement the `_controller_configs` property as done below

```python
class Panda(BaseAgent):
    # ...
    arm_joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    gripper_joint_names = [
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100

    @property
    def _controller_configs(self):
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=-0.01,  # a trick to have force when the object is thin
            upper=0.04,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_joint_pos=dict(
              arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos
            ),
        )
        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)
```

We defined two controllers to control the arm joints and one for the gripper. Using a dictionary, you can define multiple control modes that interchangeably use different controllers of the joints. Above we defined a `pd_joint_delta_pos` and a `pd_joint_pos` controller which switch just the controller of the arm joints.

Stiffness corresponds with the P and damping corresponds with the D of PD controllers, see the [controllers page](../concepts/controllers.md#terminology) for more details


## Advanced Tips and Tricks:

### Mobile Bases

Robots like Fetch have a mobile base, which allows translational movement and rotational movement of the entire robot. In simulation, it is not trivial to simulate the actual physics of wheels moving along a floor and simulating this would be fairly slow. 

Instead, similar to many other simulators a "fake" mobile base is made (that is realistic enough to easily do sim2real transfer in terms of the controller). This is made by modifying a URDF of a robot like Fetch, and adding joints that let the base link translate (prismatic joint) and rotate (revolute joint). 

### Tactile Sensing

WIP

For now see the implementation of [Allegro hand with touch sensors](https://github.com/haosulab/ManiSkill2/blob/dev/mani_skill/agents/robots/allegro_hand/allegro_touch.py)

### Quadrupeds / Legged motion

WIP