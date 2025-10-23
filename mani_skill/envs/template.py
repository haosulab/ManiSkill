"""
Code for a minimal environment/task with just a robot being loaded. We recommend copying this template and modifying as you need.

At a high-level, ManiSkill tasks can minimally be defined by what agents/actors are
loaded, how agents/actors are randomly initialized during env resets, how goals are randomized and parameterized in observations, and success conditions

Environment reset is comprised of running two functions, `self._reconfigure` and `self.initialize_episode`, which is auto
run by ManiSkill. As a user, you can override a number of functions that affect reconfiguration and episode initialization.

Reconfiguration will reset the entire environment scene and allow you to load/swap assets and agents.

Episode initialization will reset the poses of all actors, articulations, and agents,
in addition to initializing any task relevant data like a goal

See comments for how to make your own environment and what each required function should do. If followed correctly you can easily build a
task that can simulate on the CPU and be parallelized on the GPU without having to manage GPU memory and parallelization apart from some
code that need to be written in batched mode (e.g. reward, success conditions)

For a minimal implementation of a simple task, check out
mani_skill /envs/tasks/push_cube.py which is annotated with comments to explain how it is implemented
"""


from typing import Any, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


# register the environment by a unique ID and specify a max time limit. Now once this file is imported you can do gym.make("CustomEnv-v0")
@register_env("CustomEnv-v1", max_episode_steps=200)
class CustomEnv(BaseEnv):
    """
    Task Description
    ----------------
    Add a task description here

    Randomizations
    --------------
    - how is it randomized?
    - how is that randomized?

    Success Conditions
    ------------------
    - what is done to check if this task is solved?

    Visualization: link to a video/gif of the task being solved
    """

    # here you can define a list of robots that this task is built to support and be solved by. This is so that
    # users won't be permitted to use robots not predefined here. If SUPPORTED_ROBOTS is not defined then users can do anything
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    # if you want to say you support multiple robots you can use SUPPORTED_ROBOTS = [["panda", "panda"], ["panda", "fetch"]] etc.

    # to help with programming, you can assert what type of agents are supported like below, and any shared properties of self.agent
    # become available to typecheckers and auto-completion. E.g. Panda and Fetch both share a property called .tcp (tool center point).
    agent: Union[Panda, Fetch]
    # if you want to do typing for multi-agent setups, use this below and specify what possible tuples of robots are permitted by typing
    # this will then populate agent.agents (list of the instantiated agents) with the right typing
    # agent: MultiAgent[Union[Tuple[Panda, Panda], Tuple[Panda, Panda, Panda]]]

    # in the __init__ function you can pick a default robot your task should use e.g. the panda robot by setting a default for robot_uids argument
    # note that if robot_uids is a list of robot uids, then we treat it as a multi-agent setup and load each robot separately.
    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # Specify default simulation/gpu memory configurations. Note that tasks need to tune their GPU memory configurations accordingly
    # in order to save memory while also running with no errors. In general you can start with low values and increase them
    # depending on the messages that show up when you try to run more environments in parallel. Since this is a python property
    # you can also check self.num_envs to dynamically set configurations as well
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    """
    Reconfiguration Code

    below are all functions involved in reconfiguration during environment reset called in the same order. As a user
    you can change these however you want for your desired task. These functions will only ever be called once in general. In CPU simulation,
    for some tasks these may need to be called multiple times if you need to swap out object assets. In GPU simulation these will only ever be called once.
    """

    def _load_agent(self, options: dict):
        # this code loads the agent into the current scene. You should use it to specify the initial pose(s) of the agent(s)
        # such that they don't collide with other objects initially
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0]))

    def _load_scene(self, options: dict):
        # here you add various objects like actors and articulations. If your task was to push a ball, you may add a dynamic sphere object on the ground
        pass

    @property
    def _default_sensor_configs(self):
        # To customize the sensors that capture images/pointclouds for the environment observations,
        # simply define a CameraConfig as done below for Camera sensors. You can add multiple sensors by returning a list
        pose = sapien_utils.look_at(
            eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1]
        )  # sapien_utils.look_at is a utility to get the pose of a camera that looks at a target

        # to see what all the sensors capture in the environment for observations, run env.render_sensors() which returns an rgb array you can visualize
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        # this is just like _sensor_configs, but for adding cameras used for rendering when you call env.render()
        # when render_mode="rgb_array" or env.render_rgb_array()
        # Another feature here is that if there is a camera called render_camera, this is the default view shown initially when a GUI is opened
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return [CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)]

    def _setup_sensors(self, options: dict):
        # default code here will setup all sensors. You can add additional code to change the sensors e.g.
        # if you want to randomize camera positions
        return super()._setup_sensors()

    def _load_lighting(self, options: dict):
        # default code here will setup all lighting. You can add additional code to change the lighting e.g.
        # if you want to randomize lighting in the scene
        return super()._load_lighting()

    """
    Episode Initialization Code

    below are all functions involved in episode initialization during environment reset called in the same order. As a user
    you can change these however you want for your desired task. Note that these functions are given a env_idx variable.

    `env_idx` is a torch Tensor representing the indices of the parallel environments that are being initialized/reset. This is used
    to support partial resets where some parallel envs might be reset while others are still running (useful for faster RL and evaluation).
    Generally you only need to really use it to determine batch sizes via len(env_idx). ManiSkill helps handle internally a lot of masking
    you might normally need to do when working with GPU simulation. For specific details check out the push_cube.py code
    """

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        pass

    """
    Modifying observations, goal parameterization, and success conditions for your task

    the code below all impact some part of `self.step` function
    """

    def evaluate(self, obs: Any):
        # this function is used primarily to determine success and failure of a task, both of which are optional. If a dictionary is returned
        # containing "success": bool array indicating if the env is in success state or not, that is used as the terminated variable returned by
        # self.step. Likewise if it contains "fail": bool array indicating the opposite (failure state or not) the same occurs. If both are given
        # then a logical OR is taken so terminated = success | fail. If neither are given, terminated is always all False.
        #
        # You may also include additional keys which will populate the info object returned by self.step and that will be given to
        # `_get_obs_extra` and `_compute_dense_reward`. Note that as everything is batched, you must return a batched array of
        # `self.num_envs` booleans (or 0/1 values) for success an dfail as done in the example below
        return {
            "success": torch.zeros(self.num_envs, device=self.device, dtype=bool),
            "fail": torch.zeros(self.num_envs, device=self.device, dtype=bool),
        }

    def _get_obs_extra(self, info: dict):
        # should return an dict of additional observation data for your tasks
        # this will be included as part of the observation in the "extra" key when obs_mode="state_dict" or any of the visual obs_modes
        # and included as part of a flattened observation when obs_mode="state". Moreover, you have access to the info object
        # which is generated by the `evaluate` function above
        return dict()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # you can optionally provide a dense reward function by returning a scalar value here. This is used when reward_mode="dense"
        # note that as everything is batched, you must return a batch of of self.num_envs rewards as done in the example below.
        # Moreover, you have access to the info object which is generated by the `evaluate` function above
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

    def get_state_dict(self):
        # this function is important in order to allow accurate replaying of trajectories. Make sure to specify any
        # non simulation state related data such as a random 3D goal position you generated
        # alternatively you can skip this part if the environment's rewards, observations, eval etc. are dependent on simulation data only
        # e.g. self.your_custom_actor.pose.p will always give you your actor's 3D position
        state = super().get_state_dict()
        # state["goal_pos"] = add_your_non_sim_state_data_here
        return state

    def set_state_dict(self, state):
        # this function complements get_state and sets any non simulation state related data correctly so the environment behaves
        # the exact same in terms of output rewards, observations, success etc. should you reset state to a given state and take the same actions
        self.goal_pos = state["goal_pos"]
        super().set_state_dict(state)
