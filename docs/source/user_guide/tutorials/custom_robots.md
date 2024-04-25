# Custom Robots

ManiSkill supports importing robots and assets via URDF and MJCF definitions. As ManiSkill was designed to allow one to flexibly change mounted sensors, controllers, and robot embodiments, ManiSkill at the moment does not automatically import those elements into the created robot, they must be defined by you.

In summary, the following elements must be completed before a robot is usable.

1. Create a robot class and specify a uid, as well as a urdf/mjcf file to import
2. Define robot controller(s) (e.g. PD joint position control)
3. (Optional): Define mounted sensors / sensors relative to the robot (e.g. wrist cameras)
4. (Optional): Define useful keyframes (snapshots of robot state) for users

This tutorial will guide you through on how to implement the Panda robot in ManiSkill. It will also cover some tips/tricks for modelling other categories of robots to ensure the simulation runs fast and accurately (e.g., mobile manipulators like Fetch, quadrupeds like Anymal)


## 1. Robot Class and Importing the Robot

To create your own robot (also known as an Agent) you need to inherit the `BaseAgent` class, give it name, and optionally register the agent.

```python
import sapien
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.registration import register_agent
@register_agent()
class MyPanda(BaseAgent):
    uid = "my_panda"
```

Registering the agent allows you to create environments that instantiate your robot for you via a string uid in the future with the code below:

```python
env = gym.make("EmptyEnv-v1", robot_uids="my_panda")
```

With the base robot class started, now we need to import a robot definition from a URDF or MJCF

### Importing the Robot

To import a URDF/MJCF file, you simply provide a path to the definition file and ManiSkill handles the importing. ManiSkill will automatically create the appropriate articulation with links and joints defined in the given definition file. Internally ManiSkill uses either the URDFLoader or MJCFLoader class in the `mani_skill.utils.building` module.

<!-- TODO (stao): provide some demo script / checkpointer code to let user visualize progress in tutorial -->

#### URDF

To get started, you first need to get a valid URDF file like this one for the Panda robot: https://github.com/haosulab/ManiSkill/blob/main/mani_skill/assets/robots/panda/panda_v2.urdf

Then in the agent class add

```python
class MyPanda(BaseAgent):
    uid = "my_panda"
    urdf_path = f"path/to/your/robot.urdf"
```

Note that there are a number of common issues users may face (often due to incorrectly formatted URDFs / collision meshes) which are documented in the [FAQ / Troubleshooting Section](#faq--troubleshooting)


<!-- For a starting template check out https://github.com/haosulab/ManiSkill/blob/main/mani_skill/agents/robots/_template/template_robot.py -->

#### Mujoco MJCF

ManiSkill supports importing [Mujoco's MJCF format](https://mujoco.readthedocs.io/en/latest/modeling.html) of files to load robots (and other objects), although not all features are supported.

For example code that loads the robot and the scene see https://github.com/haosulab/ManiSkill/blob/main/mani_skill/envs/tasks/control/cartpole.py


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

Stiffness corresponds with the P and damping corresponds with the D of PD controllers, see the [controllers page](../concepts/controllers.md#terminology) for more details.

Tuning the values of stiffness, damping, and other properties affect the sim2real transfer of a simulated robot to the real world. At the moment our team is working on developing a better pipeline with documentation for system identification to pick better controllers and/or hyperparameters. 

Note that when taking a robot implemented in another simulator like Mujoco, you usually cannot directly copy the joint hyperparameters to ManiSkill, so you almost always need some manual tuning. 

## 3. Defining Sensors

ManiSkill supports defining sensors mounted onto the robot and sensors positioned relative to the robot by defining the `_default_sensor_configs` property.

An example of this done in the Panda robot with a real sense camera attached:

```python
from mani_skill.sensors.camera import CameraConfig
class PandaRealSensed435(Panda):
    # ...
    @property
    def _default_sensor_configs(self):
        return [
            CameraConfig(
                uid="hand_camera",
                pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link"],
            )
        ]
```

You simply return a sensor config (here we use a CameraConfig) to define the sensor to add, and specify where to mount the sensor. For most sensors, you must define a pose, which is now used as a pose relative to the mount pose. In the example above we add a camera to the camera link / wrist mount of the panda robot (which is already oriented facing the correct direction so the pose defined is just the identity)

## 4. Defining Keyframes

It sometimes is useful to define some predefined robot poses and joint positions that users can initialize to to visualize the robot in poses of interest. This is an idea adopted from [Mujoco's keyframes](https://mujoco.readthedocs.io/en/stable/XMLreference.html#keyframe)

For example, we define a "standing" keyframe for the Unitree H1 robot like so

```python
from mani_skill.agents.base_agent import BaseAgent, Keyframe
# ...
class UnitreeH1(BaseAgent):
    # ...
    keyframes = dict(
        standing=Keyframe(
            pose=sapien.Pose(p=[0, 0, 0.975]),
            qpos=np.array([0, 0, 0, 0, 0, 0, 0, -0.4, -0.4, 0.0, 0.0, 0.8, 0.8, 0.0, 0.0, -0.4, -0.4, 0.0, 0.0]) * 1,
        )
    )
```

The keyframe can also specify `qvel` values as well. Using that keyframe you can set the robot to the given pose, qpos, qvel and you can get the desired predefined keyframe

:::{figure} images/unitree_h1_standing.png 
:::

## Advanced Tips and Tricks:

### Mobile Bases

Robots like Fetch have a mobile base, which allows translational movement and rotational movement of the entire robot. In simulation, it is not trivial to simulate the actual physics of wheels moving along a floor and simulating this would be fairly slow. 

Instead, similar to many other simulators a "fake" mobile base is made (that is realistic enough to easily do sim2real transfer in terms of the controller). This is made by modifying a URDF of a robot like Fetch, and adding joints that let the base link translate (prismatic joint) and rotate (revolute joint). 

### Tactile Sensing

WIP

For now see the implementation of [Allegro hand with touch sensors](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/agents/robots/allegro_hand/allegro_touch.py)

### Quadrupeds / Legged motion

WIP

## FAQ / Troubleshooting

### On Importing URDF files

**Loaded robot does not have the right render materials / colors showing up:**
Likely caused by improper use of `<material>` tags in the URDF. Double check the material tags each have unique names and are correctly written according to the URDF format

**The collision of the robot seems off (e.g. sinks through floor, objects that should collide are not colliding etc.):**

In the viewer when visualizing the robot you created, click any link on the robot and under the Articulation tab scroll down and click Show collision. This visualizes all collision meshes used for contact simulation and shows you what was loaded from the URDF. You can then edit / modify the `<collision>` tags of the URDF accordingly

**The collision shape looks completely different from the visual (like a convex version of it)**

This can be caused by a few reasons. One may be that your defined base agent has its `load_multiple_collisions` property set to False. If the collision meshes you use have multiple convex shapes that can be loaded (preferrably a .ply or .glb format), then setting `load_multiple_collisions = True` in your custom robot class can work.

Another reason is if your collision mesh is in the .stl format. Our loader has some issues loading .stl files at times and we recommend converting them to `.glb` as that is the easiest for our system to load and interpret. 

Another issue is if your collision mesh does not have multiple convex shapes, you may have to decompose those meshes yourself via a tool like COACD.
<!-- TODO (stao): Detail a pipeline to semi-automatically do this -->