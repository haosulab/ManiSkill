"""
Code for a minimal environment/task with just a robot being loaded. We recommend copying this template and modifying as you need.

At a high-level, ManiSkill2 tasks can minimally be defined by how the environment resets, what agents/objects are 
loaded, goal parameterization, and success conditions

Environment reset is comprised of running two functions, `self.reconfigure` and `self.initialize_episode`, which is auto
run by ManiSkill2. As a user, you can override a number of functions that affect reconfiguration and episode initialization.

Reconfiguration will reset the entire environment scene and allow you to load/swap assets and agents.

Episode initialization will reset the positions of all objects (called actors), articulations, and agents,
in addition to initializing any task relevant data like a goal

See comments for how to make your own environment and what each required function should do
"""

from collections import OrderedDict
from typing import Any, Dict, Type

import numpy as np
import sapien
import sapien.physx as physx

from mani_skill2.agents.base_agent import BaseAgent
from mani_skill2.agents.robots import (
    ROBOTS,  # a dictionary mapping robot name to robot class that inherits BaseAgent
)
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.sapien_utils import (  # import various useful utilities for working with sapien
    get_obj_by_name,
    look_at,
)


class CustomEnv(BaseEnv):
    # in the __init__ function you can pick a default robot your task should use e.g. the panda robot
    def __init__(self, *args, robot_uid="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uid=robot_uid, **kwargs)

    """
    One time configuration code
    """

    def _register_sensors(self):
        # To customize the sensors that capture images/pointclouds for the environment observations,
        # simply define a CameraConfig as done below for Camera sensors. You can add multiple sensors by returning a list
        pose = look_at(
            eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1]
        )  # look_at is a utility to get the pose of a camera that looks at a target
        return [
            CameraConfig("base_camera", pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10)
        ]

    def _register_render_cameras(self):
        # this is just like _register_sensors, but for adding cameras used for rendering when you call env.render()
        if self.robot_uid == "panda":
            pose = look_at(eye=[0.4, 0.4, 0.8], target=[0.0, 0.0, 0.4])
        else:
            pose = look_at(eye=[0.5, 0.5, 1.0], target=[0.0, 0.0, 0.5])
        return [CameraConfig("render_camera", pose.p, pose.q, 512, 512, 1, 0.01, 10)]

    def _setup_viewer(self):
        # you add code after calling super()._setup_viewer() to configure how the SAPIEN viewer (a GUI) looks
        super()._setup_viewer()
        self._viewer.set_camera_xyz(0.8, 0, 1.0)
        self._viewer.set_camera_rpy(0, -0.5, 3.14)

    """
    Episode Initialization Code

    below are all functions involved in episode initialization during environment reset called in the same order. As a user
    you can change these however you want for your desired task.
    """

    def _initialize_agent(self):
        # here you initialize the agent/robot. This usually involves setting the joint position of the robot. We provide
        # some default code below for panda and xmate3 set the robot to a "rest" position with a little bit of noise for randomization

        if self.robot_uid == "panda":
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            qpos[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.robot_uid == "xmate3_robotiq":
            qpos = np.array(
                [0, np.pi / 6, 0, np.pi / 3, 0, np.pi / 2, -np.pi / 2, 0, 0]
            )
            qpos[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.562, 0, 0]))
        else:
            raise NotImplementedError(self.robot_uid)

    def _initialize_actors(self):
        pass

    def _initialize_articulations(self):
        pass

    def _initialize_task(self):
        # we highly recommend to generate some kind of "goal" information to then later include in observations
        # goal can be parameterized as a state (e.g. target pose of a object)
        pass

    """
    Reconfiguration Code

    below are all functions involved in reconfiguration during environment reset called in the same order. As a user
    you can change these however you want for your desired task.
    """

    def _load_agent(self):
        # this code loads the agent into the current scene. You can usually ignore this function by deleting it or calling the inherited
        # BaseEnv._load_agent function
        super()._load_agent()

    def _load_actors(self):
        # here you add various objects (called actors). If your task was to push a ball, you may add a dynamic sphere object on the ground
        pass

    def _load_articulations(self):
        # here you add various articulations. If your task was to open a drawer, you may add a drawer articulation into the scene
        pass

    def _setup_sensors(self):
        # default code here will setup all sensors. You can add additional code to change the sensors e.g.
        # if you want to randomize camera positions
        return super()._setup_sensors()

    def _setup_lighting(self):
        # default code here will setup all lighting. You can add additional code to change the lighting e.g.
        # if you want to randomize lighting in the scene
        return super()._setup_lighting()

    """
    Modifying observations, goal parameterization, and success conditions for your task

    the code below all impact some part of `self.step` function
    """

    def _get_obs_extra(self):
        # should return an OrderedDict of additional observation data for your tasks
        # this will be included as part of the observation in the "extra" key
        return OrderedDict()

    def evaluate(self):
        # should return a dictionary containing "success": bool indicating if the environment is in success state or not. The value here is also what the sparse reward is
        # for the task. You may also include additional keys which will populate the info object returned by self.step
        return {"success": False}

    def compute_dense_reward(self, obs: Any, action: np.ndarray, info: Dict):
        # you can optionally provide a dense reward function by returning a scalar value here. This is used when reward_mode="dense"
        return 0

    def compute_normalized_dense_reward(self, obs: Any, action: np.ndarray, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
