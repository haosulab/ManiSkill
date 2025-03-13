"""
Gymnasium environment interface for controlling a real robot.

The setup of this environment is the same as the ManiSkill BaseEnv in terms of observation and action spaces, and uses the same
controller code for the real robot as used in simulation.

Note that as many operations of simulation environments are not available for real environments, we do not inherit from BaseEnv and simply
reference some of the BaseEnv functions for consistency instead.

One small difference as well between RealEnv and BaseEnv is that the code for fetching raw real world sensor data is robot dependent, not environment dependent.

So for real world deployments you may take an existing implementation of a real robot class and use it as a starting point for your own implementation to add e.g. more cameras
or generate other kinds of sensor data.
"""
from typing import Any, Callable, Dict, List, Optional

import gymnasium as gym
import torch

from mani_skill.agents.base_real_agent import BaseRealAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common


class Sim2RealEnv(gym.Env):
    """
    Sim2RealEnv is a class that lets you interface with a real robot and align the real robot and environment with a simulation environment. It tries to ensure the action and observation space
    are the exact same in the real and simulation environments. Any wrappers you apply to the simulation environment are also used in the Sim2RealEnv automatically.

    There are some caveats in which you may need to override this class / write your own code instead:

    - If you use privileged features in the simulation environment like an object's pose then we cannot retrieve those poses in the real environment.
    You can for example override the `_get_obs_extra` function to compute those values in the real environment via a perception pipeline.

    - While we align controllers and observation shapes/ordering as much as possible, there can
    still be distribution shifts between the simulation and real environment. These can include vision gaps (sim images looking not like the real world)
    and sensor biases and noise.

    Args:
        sim_env (BaseEnv): The simulation environment that the real environment should be aligned with.
        agent (BaseRealAgent): The real robot agent to control. This must be an object that inherits from BaseRealAgent.
        obs_mode (str): The observation mode to use.
        real_reset_function (Optional[Callable[[Sim2RealEnv, Optional[int], Optional[dict]], None]]): The function to call to reset the real robot. By default this is None and we use a default reset function which
            calls the simulation reset function and resets the agent/robot qpos to whatever the simulation reset function sampled. This function is given access to the Sim2RealEnv instance, the given seed and options dictionary
            similar to a standard gym reset function.
    """

    def __init__(
        self,
        sim_env: BaseEnv,
        agent: BaseRealAgent,
        obs_mode: str = "rgb",
        real_reset_function: Optional[
            Callable[["Sim2RealEnv", Optional[int], Optional[dict]], None]
        ] = None,
        # obs_mode: Optional[str] = None,
        reward_mode: Optional[str] = None,
        # control_mode: Optional[str] = None,
        # render_mode: Optional[str] = None,
        # robot_uids: BaseRealAgent = None,
    ):
        self.sim_env = sim_env
        self._base_sim_env: BaseEnv = sim_env.unwrapped

        self._reward_mode = reward_mode
        self._obs_mode = obs_mode

        # setup spaces
        self._orig_single_action_space = self._base_sim_env._orig_single_action_space
        self.action_space = self.sim_env.action_space
        self.observation_space = self.sim_env.observation_space

        # setup step and reset functions and handle wrappers for the user

        def default_real_reset_function(self: Sim2RealEnv, seed=None, options=None):
            # self.agent.reset(qpos=self.sim_env.agent.robot.qpos.cpu())  # TODO (stao): re-enable this
            input("Press enter if the environment is reset")

        self.real_reset_function = real_reset_function or default_real_reset_function

        class RealEnvStepReset:
            def step(dummy_self, action):
                ret = BaseEnv.step(self, action)
                return ret

            def reset(dummy_self, seed=None, options=None):
                self.real_reset_function(self, seed, options)
                # reset controller/agent
                return BaseEnv.get_obs(self), {}

        cur_env = self.sim_env
        wrappers: List[gym.Wrapper] = []
        while isinstance(cur_env, gym.Wrapper):
            wrappers.append(cur_env)
            cur_env = cur_env.env

        self._handle_wrappers = len(wrappers) > 0
        if self._handle_wrappers:
            self._first_wrapper = wrappers[0]
            self._last_wrapper = wrappers[-1]

        self._env_with_real_step_reset = RealEnvStepReset()
        """a simple object that defines the real step/reset functions for gym wrappers to call and use."""

        self._sensor_names = list(self._base_sim_env.scene.sensors.keys())
        """list of sensors the simulation environment uses"""

        # setup the real agent based on the simulation agent
        self.agent = agent
        self.agent._sim_agent = self._base_sim_env.agent
        # TODO create real controller class based on sim one?? Or can we just fake the data
        self.agent._sim_agent.controller.qpos

    def _step_action(self, action):
        """Re-implementation of the simulated BaseEnv._step_action function for real environments. This uses the simulation agent's
        controller to compute the joint targets/velocities without stepping the simulator"""
        # BaseEnv._step_action(self, action)
        if action.shape == self._orig_single_action_space.shape:
            action = common.batch(action)
            action = common.to_tensor(action)
        self._base_sim_env.agent.set_action(action)
        # self.sim_env.agent.controller
        self.agent.set_target_qpos(action)

    def step(self, action):
        """
        In order to make users able to use most gym environment wrappers without having to write extra code for the real environment
        we temporarily swap the last wrapper's .env property with the RealEnvStepReset environment that has the real step/reset functions
        """
        if self._handle_wrappers:
            orig_env = self._last_wrapper.env
            self._last_wrapper.env = self._env_with_real_step_reset
            ret = self._first_wrapper.step(action)
            self._last_wrapper.env = orig_env
            return ret
        else:
            return self._env_with_real_step_reset.step(action)

    def reset(self, seed=None, options=None):
        if self._handle_wrappers:
            orig_env = self._last_wrapper.env
            self._last_wrapper.env = self._env_with_real_step_reset
            ret = self._first_wrapper.reset(seed=seed, options=options)
            self._last_wrapper.env = orig_env
            return ret
        else:
            return self._env_with_real_step_reset.reset(seed, options)

    # -------------------------------------------------------------------------- #
    # reimplementations of simulation BaseEnv observation related functions
    # -------------------------------------------------------------------------- #
    def _get_obs_agent(self):
        # using the original user implemented sim env's _get_obs_agent function in case they modify it e.g. to remove qvel values as they might be too noisy
        return self._base_sim_env.__class__._get_obs_agent(self)

    def _get_obs_extra(self, info: Dict):
        # using the original user implemented sim env's _get_obs_extra function in case they modify it e.g. to include engineered features like the tcp_pose of the robot
        try:
            return self._base_sim_env.__class__._get_obs_extra(self, info)
        except:
            # Print the original error
            import traceback

            print(f"Error in _get_obs_extra: {traceback.format_exc()}")

            # Print another message
            print(
                "If there is an error above a common cause is that the _get_obs_extra function defined in the simulation environment is using information not available in the real environment or real agent."
                "In this case you can override the _get_obs_extra function in the Sim2RealEnv class to compute the desired information in the real environment via a e.g., perception pipeline."
            )
            exit(-1)

    def _get_obs_sensor_data(self, apply_texture_transforms: bool = True):
        # note apply_texture_transforms is not used for real envs, data is expected to already be transformed to standard texture names, types, and shapes.
        self.agent.capture_sensor_data(self._sensor_names)
        data = self.agent.get_sensor_obs(self._sensor_names)
        return data

    def _get_obs_with_sensor_data(
        self, info: Dict, apply_texture_transforms: bool = True
    ) -> dict:
        """Get the observation with sensor data"""
        return BaseEnv._get_obs_with_sensor_data(self, info, apply_texture_transforms)

    def get_sensor_params(self):
        return self.agent.get_sensor_params(self._sensor_names)

    def get_info(self):
        return (
            {}
        )  # TODO (stao): add elapsed steps and other things? document how to write real world success function?
        # return BaseEnv.get_info(self)

    # TODO (stao): add real world render function for episode recording
    # def render(self):
    #     return BaseEnv.render(self)

    # -------------------------------------------------------------------------- #
    # reimplementations of simulation BaseEnv reward related functions. By default you can leave this alone but if you do want to
    # support computing rewards in the real world you can override these functions.
    # -------------------------------------------------------------------------- #
    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """
        Computes the sparse reward. By default this function tries to use the success/fail information in
        returned by the evaluate function and gives +1 if success, -1 if fail, 0 otherwise"""
        return BaseEnv.compute_sparse_reward(self, obs, action, info)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        raise NotImplementedError()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        raise NotImplementedError()
