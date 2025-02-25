"""base environment for real robots, uses controller from reference simulation agent"""
import copy
import time
from functools import cached_property
from typing import Dict, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector.utils import batch_space

# ManiSkill imports
from mani_skill.utils import common, gym_utils


# control frequency automatically scheduled based on sim_env_id
# robot controller is used from reference sim_env_id provided in subclass
# TODO (xhin): abstract robot attribubte to robot class similar to ms3 baserobot class
class BaseRealEnv(gym.Env):
    """
    # once the physical robot and configs are set up, allows easy testing of evaluation environment
    # create image transform for raw camera outputs as necessary:
    real_robot = ...
    real_env = gym.make("RealGrabCube-v1", real_robot=real_robot, control_mode="pd_joint_delta_pos")
    real_env = FlattenRGBDObservationWrapper(real_env, rgb=True, depth=False, state=True)

    # BaseRealEnv use pseudocode
    # stepping timing already configured, no need to manually sleep during evaluation:
    obs, info = real_env.reset() # resets robot to provided keyframe
    for _ in range(num_steps):
        action = policy.get_action(obs["rgb"], obs["state"])
        obs, _, _, _, _ = real_env.step(action) # action to raw control done by BaseRealEnv, using sim_env controller

    Warning: safety of robot motors may require the following:
        Correct calibration and setup of the robot
        Qpos limits correctly defined in the URDF
        ManiSkill Controller limiting per step change in qpos (regardless of controller type)
        Human intervention and early stopping if policy causes unwanted robot collisions (e.g. with table)
    """

    agent: None  # To be filled out in subclasses
    # TODO (xhin): neaten typing
    def __init__(self, real_agent, keyframe_id, control_timing=True):
        self.agent = real_agent
        self.sim_agent = self.agent.sim_agent
        self.keyframe_id = keyframe_id
        self.control_timing = control_timing

        # extract sim_agent info to use on real robot
        self.control_mode = self.agent.control_mode
        self.control_freq = self.agent.control_freq
        self.single_action_space = self.sim_agent.single_action_space
        self._orig_single_action_space = copy.deepcopy(self.single_action_space)

        # TODO (xhin): controller interpolate sets drive targets but doesn't change self._target_qpos at each step per control step
        # TODO (xhin): also, it doesn't currently work with use_target at all
        if self.agent.controller.config.interpolate:
            raise NotImplementedError(
                "Controller with interpolation not impemented for real robot yet"
            )

        self.robot_keyframe_qpos = torch.from_numpy(
            self.sim_agent.keyframes[keyframe_id].qpos
        )
        self.elapsed_steps = 0

        # hardcode cpu as we're running single real environment
        self.device = torch.device("cpu")
        self.num_envs = 1

        # set necessary gym attributes
        self.sim_agent.controller.reset()
        obs = self.get_obs()
        self._init_raw_obs = common.to_cpu_tensor(obs)
        # initialize the cached properties
        self.single_observation_space
        self.observation_space

        qpos_limits = self.agent.controller.articulation.get_qlimits()[
            0, self.agent.controller.active_joint_indices
        ].cpu()
        self.qpos_lower_limits = qpos_limits[:, 0]
        self.qpos_upper_limits = qpos_limits[:, 1]

    def reset(self, seed=0, options=None):
        # user's implementation for robot reset
        self.agent.reset(self.robot_keyframe_qpos)

        # manually ensure controller is reset at start = no unintentional large actions
        self.agent.controller.reset()
        self.sim_agent.controllers[self.agent.control_mode].reset()
        self.sim_agent.controllers[
            self.agent.control_mode
        ]._target_qpos = self.agent.qpos.unsqueeze(0)

        obs = self.get_obs()
        self.elapsed_steps = 0
        return obs, dict(reconfigure=False)

    def _get_obs_agent(self):
        # return dict(qpos=common.batch(self.agent.qpos))
        return dict()

    def _get_obs_extra(self):
        return dict()

    # TODO (xhin): test running with multiple cameras
    def _get_obs_sensor_data(self):
        return self.agent.get_obs_sensor_data()

    def _get_obs_with_sensor_data(self):
        return dict(
            agent=self._get_obs_agent(),
            extra=self._get_obs_extra(),
            sensor_param=dict(),
            sensor_data=self._get_obs_sensor_data(),
        )

    def get_obs(self):
        return self._get_obs_with_sensor_data()

    def _step_action(
        self, action: Union[None, np.ndarray, torch.Tensor, Dict]
    ) -> Union[None, torch.Tensor]:
        set_action = False
        action_is_unbatched = False
        if action is None:  # simulation without action
            pass
        elif isinstance(action, np.ndarray) or isinstance(action, torch.Tensor):
            action = common.to_tensor(action, device=self.device)
            if action.shape == self._orig_single_action_space.shape:
                action_is_unbatched = True
            set_action = True
        else:
            raise TypeError(type(action))

        # automatic step timing
        if self.elapsed_steps == 0:
            self.episode_start_time = time.perf_counter()
        elif self.control_timing:
            control_step_time = (
                self.episode_start_time + (1 / self.control_freq) * self.elapsed_steps
            )
            curr_time = time.perf_counter()
            # curr_time is ideally always < control_step_time
            if curr_time > control_step_time:
                print(
                    f"Warning: Control step late by {curr_time - control_step_time} seconds. Consider reducing env control_frequency"
                )
                # move schedule to account for discrepancy
                self.episode_start_time += curr_time - control_step_time
            else:
                time.sleep(control_step_time - curr_time)

        # send robot action
        if set_action:
            if action_is_unbatched:
                action = common.batch(action)
            assert (
                action.shape[0] == 1
            ), f"real robot cannot process batched input, got {action.shape}"
            # regardless of controller type, MS3 controller output is global joint positions
            self.agent.controller.set_action(action)
            target_qpos = self.agent.controller._target_qpos.clone()
            # clip the global qpos to min and max qpos from robot description file
            target_qpos = target_qpos.clip(
                min=self.qpos_lower_limits, max=self.qpos_upper_limits
            )
            self.agent.send_qpos(target_qpos[0])

    def step(self, action):
        self._step_action(action)
        self.elapsed_steps += 1
        obs = self.get_obs()
        return (
            obs,
            None,  # reward
            False,  # terminated
            False,  # truncated
            dict(),  # info
        )

    def render(self):
        return self.agent.render()

    ############ Supporting functions for MS3 gym wrappers ############
    def update_obs_space(self, obs: torch.Tensor):
        self._init_raw_obs = obs
        del self.single_observation_space
        del self.observation_space
        self.single_observation_space
        self.observation_space

    @cached_property
    def single_observation_space(self) -> gym.Space:
        """the unbatched observation space of the environment"""
        return gym_utils.convert_observation_to_space(
            common.to_numpy(self._init_raw_obs), unbatched=True
        )

    @cached_property
    def observation_space(self) -> gym.Space:
        """the batched observation space of the environment"""
        return batch_space(self.single_observation_space, n=self.num_envs)

    def close(self):
        self.robot.disconnect()
        print("Robot disconnected")
