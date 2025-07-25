import time
from typing import Any, Callable, Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch

from mani_skill.agents.base_real_agent import BaseRealAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import Camera, CameraConfig
from mani_skill.utils import common
from mani_skill.utils.logging_utils import logger


class Sim2RealEnv(gym.Env):
    """
    Sim2RealEnv is a class that lets you interface with a real robot and align the real robot and environment with a simulation environment. It tries to ensure the action and observation space
    are the exact same in the real and simulation environments. Any wrappers you apply to the simulation environment are also used in the Sim2RealEnv automatically.

    There are some caveats in which you may need to override this class / write your own code instead:

    - If you use privileged features in the simulation environment like an object's pose then we cannot retrieve those poses in the real environment. You can for example override the `_get_obs_extra` function to compute those values in the real environment via a perception pipeline.

    - While we align controllers and observation shapes/ordering as much as possible, there can still be distribution shifts between the simulation and real environment. These can include vision gaps (sim images looking not like the real world) and sensor biases and noise.

    Args:
        sim_env (BaseEnv): The simulation environment that the real environment should be aligned with.
        agent (BaseRealAgent): The real robot agent to control. This must be an object that inherits from BaseRealAgent.
        real_reset_function (Optional[Callable[[Sim2RealEnv, Optional[int], Optional[dict]], None]]): The function to call to reset the real robot. By default this is None and we use a default reset function which
            calls the simulation reset function and resets the agent/robot qpos to whatever the simulation reset function sampled, then prompts the user to press enter before continuing running.
            This function is given access to the Sim2RealEnv instance, the given seed and options dictionary similar to a standard gym reset function. The default function and example is shown below:

            .. code-block:: python

                def real_reset_function(self, seed=None, options=None):
                    self.sim_env.reset(seed=seed, options=options)
                    self.agent.reset(qpos=self.base_sim_env.agent.robot.qpos.cpu().flatten())
                    input("Press enter if the environment is reset")

        sensor_data_preprocessing_function (Optional[Callable[[Dict], Dict]]): The function to call to process the sensor data returned by the BaseRealAgent.get_sensor_data function.
            By default this is None and we use a default processing function which does the following for each sensor type:
            - Camera: Perform a center crop of the real sensor image (rgb or depth) to have the same aspect ratio as the simulation sensor image. Then resize the image to the simulation sensor image shape using cv2.resize

        skip_data_checks (bool): If False, this will reset the sim and real environments once to check if observations are aligned. It is recommended
            to keep this False.
        control_freq (Optional[int]): The control frequency of the real robot. By default this is None and we use the same control frequency as the simulation environment.

    """

    metadata = {"render_modes": ["rgb_array", "sensors", "all"]}

    def __init__(
        self,
        sim_env: gym.Env,
        agent: BaseRealAgent,
        real_reset_function: Optional[
            Callable[["Sim2RealEnv", Optional[int], Optional[dict]], None]
        ] = None,
        sensor_data_preprocessing_function: Optional[Callable[[Dict], Dict]] = None,
        render_mode: Optional[str] = "sensors",
        skip_data_checks: bool = False,
        control_freq: Optional[int] = None,
    ):
        self.sim_env = sim_env
        self.num_envs = 1
        assert (
            self.sim_env.unwrapped.backend.sim_backend == "physx_cpu"
        ), "For the Sim2RealEnv we expect the simulation to be using the physx_cpu simulation backend currently in order to correctly align the robot"

        # copy over some sim parameters/settings
        self.device = self.sim_env.unwrapped.backend.device
        self.sim_freq = self.sim_env.unwrapped.sim_freq
        self.control_freq = control_freq or self.sim_env.unwrapped.control_freq

        # control timing
        self.control_dt = 1 / self.control_freq
        self.last_control_time: Optional[float] = None

        self.base_sim_env: BaseEnv = sim_env.unwrapped
        """the unwrapped simulation environment"""

        obs_mode = self.base_sim_env.obs_mode
        reward_mode = self.base_sim_env.reward_mode
        self._reward_mode = reward_mode
        self._obs_mode = obs_mode
        self.reward_mode = reward_mode
        self.obs_mode = obs_mode
        self.obs_mode_struct = self.base_sim_env.obs_mode_struct
        self.render_mode = render_mode

        self._elapsed_steps = torch.zeros((1,), dtype=torch.int32)

        # setup spaces
        self._orig_single_action_space = self.base_sim_env._orig_single_action_space
        self.action_space = self.sim_env.action_space
        self.observation_space = self.sim_env.observation_space

        # setup step and reset functions and handle wrappers for the user

        def default_real_reset_function(self: Sim2RealEnv, seed=None, options=None):
            self.sim_env.reset(seed=seed, options=options)
            self.agent.reset(qpos=self.base_sim_env.agent.robot.qpos.cpu().flatten())
            input("Press enter if the environment is reset")

        self.real_reset_function = real_reset_function or default_real_reset_function

        class RealEnvStepReset(gym.Env):
            def step(dummy_self, action):
                ret = self.base_sim_env.__class__.step(self, action)
                return ret

            def render(dummy_self):
                return self.render()

            def reset(dummy_self, seed=None, options=None):
                # TODO: reset controller/agent
                return self.get_obs(), {"reconfigure": False}

            @property
            def unwrapped(dummy_self):
                # reference the Sim2RealEnv instance
                return self

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

        self._sensor_names = list(self.base_sim_env.scene.sensors.keys())
        """list of sensors the simulation environment uses"""

        # setup the real agent based on the simulation agent
        self.agent = agent
        self.agent._sim_agent = self.base_sim_env.agent
        # TODO create real controller class based on sim one?? Or can we just fake the data
        self.agent._sim_agent.controller.qpos

        if sensor_data_preprocessing_function is not None:
            self.preprocess_sensor_data = sensor_data_preprocessing_function

        if not skip_data_checks:
            sample_sim_obs, _ = self.sim_env.reset()
            sample_real_obs, _ = self.reset()

            # perform checks to avoid errors in observation space alignment
            self._check_observations(sample_sim_obs, sample_real_obs)

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    def _step_action(self, action):
        """Re-implementation of the simulated BaseEnv._step_action function for real environments. This uses the simulation agent's
        controller to compute the joint targets/velocities without stepping the simulator"""
        action = common.to_tensor(action)
        if action.shape == self._orig_single_action_space.shape:
            action = common.batch(action)
        # NOTE (stao): this won't work for interpolated target joint position control methods at the moment
        self.base_sim_env.agent.set_action(action)

        # to best ensure whatever signals we send to the simulator robot we also send to the real robot we directly inspect
        # what drive targets the simulator controller sends and what was set by that controller on the simulated robot
        sim_articulation = self.agent.controller.articulation
        if self.last_control_time is None:
            self.last_control_time = time.perf_counter()
        else:
            dt = time.perf_counter() - self.last_control_time
            if dt < self.control_dt:
                time.sleep(self.control_dt - dt)
            else:
                logger.warning(
                    f"Control dt {self.control_dt} was not reached, actual dt was {dt}"
                )
        self.last_control_time = time.perf_counter()
        if self.agent.controller.sets_target_qpos:
            self.agent.set_target_qpos(sim_articulation.drive_targets)
        if self.agent.controller.sets_target_qvel:
            self.agent.set_target_qvel(sim_articulation.drive_velocities)

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
        else:
            ret = self._env_with_real_step_reset.step(action)
        # ensure sim agent qpos is synced
        self.base_sim_env.agent.robot.set_qpos(self.agent.robot.qpos)
        return ret

    def reset(self, seed=None, options=None):
        self.real_reset_function(self, seed, options)
        if self._handle_wrappers:
            orig_env = self._last_wrapper.env
            self._last_wrapper.env = self._env_with_real_step_reset
            ret = self._first_wrapper.reset(seed=seed, options=options)
            self._last_wrapper.env = orig_env
        else:
            ret = self._env_with_real_step_reset.reset(seed, options)
        # sets sim to whatever the real agent reset to in order to sync them. Some controllers use the agent's
        # current qpos and as this is the sim controller we copy the real world agent qpos so it behaves the same
        # moreover some properties of the robot like forward kinematic computed poses are done through the simulated robot and so qpos has to be up to date
        self.base_sim_env.agent.robot.set_qpos(self.agent.robot.qpos)
        self.agent.controller.reset()
        return ret

    # -------------------------------------------------------------------------- #
    # reimplementations of simulation BaseEnv observation related functions
    # -------------------------------------------------------------------------- #
    def get_obs(self, info=None, unflattened=False):
        # uses the original environment's get_obs function. Override this only if you want complete control over the returned observations before any wrappers are applied.
        return self.base_sim_env.__class__.get_obs(self, info, unflattened)

    def _flatten_raw_obs(self, obs: Any):
        return self.base_sim_env.__class__._flatten_raw_obs(self, obs)

    def _get_obs_agent(self):
        # using the original user implemented sim env's _get_obs_agent function in case they modify it e.g. to remove qvel values as they might be too noisy
        return self.base_sim_env.__class__._get_obs_agent(self)

    def _get_obs_extra(self, info: Dict):
        # using the original user implemented sim env's _get_obs_extra function in case they modify it e.g. to include engineered features like the tcp_pose of the robot
        try:
            return self.base_sim_env.__class__._get_obs_extra(self, info)
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
        data = self.agent.get_sensor_data(self._sensor_names)
        # observation data needs to be processed to be the same shape in simulation
        # default strategy is to do a center crop to the same shape as simulation and then resize image to the same shape as simulation
        data = self.preprocess_sensor_data(data)
        return data

    def _get_obs_with_sensor_data(
        self, info: Dict, apply_texture_transforms: bool = True
    ) -> dict:
        """Get the observation with sensor data"""
        return self.base_sim_env.__class__._get_obs_with_sensor_data(
            self, info, apply_texture_transforms
        )

    def get_sensor_params(self):
        return self.agent.get_sensor_params(self._sensor_names)

    def get_info(self):
        info = dict(elapsed_steps=self._elapsed_steps)
        return info

    # -------------------------------------------------------------------------- #
    # reimplementations of simulation BaseEnv render related functions.
    # -------------------------------------------------------------------------- #
    def render(self):
        return self.base_sim_env.__class__.render(self)

    def render_sensors(self):
        return self.base_sim_env.__class__.render_sensors(self)

    def get_sensor_images(self):
        # used by render_sensors
        obs = self._get_obs_sensor_data()
        sensor_images = dict()
        for name, sensor in self.base_sim_env.scene.sensors.items():
            if isinstance(sensor, Camera):
                sensor_images[name] = sensor.get_images(obs[name])
        return sensor_images

    # -------------------------------------------------------------------------- #
    # reimplementations of simulation BaseEnv reward related functions. By default you can leave this alone but if you do want to
    # support computing rewards in the real world you can override these functions.
    # -------------------------------------------------------------------------- #
    def get_reward(self, obs, action, info):
        return self.base_sim_env.__class__.get_reward(self, obs, action, info)

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """
        Computes the sparse reward. By default this function tries to use the success/fail information in
        returned by the evaluate function and gives +1 if success, -1 if fail, 0 otherwise"""
        return self.base_sim_env.__class__.compute_sparse_reward(
            self, obs, action, info
        )

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        raise NotImplementedError()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        raise NotImplementedError()

    # -------------------------------------------------------------------------- #
    # various checks
    # -------------------------------------------------------------------------- #
    def _check_observations(self, sample_sim_obs, sample_real_obs):
        """checks if the visual observations are aligned in terms of shape and resolution and expected data types"""

        # recursive check if the data is all the same shape
        def check_observation_match(sim_obs, real_obs, path=[]):
            """Recursively check if observations match in shape and dtype"""
            if isinstance(sim_obs, dict):
                for key in sim_obs.keys():
                    if key not in real_obs:
                        raise KeyError(
                            f"Key obs[\"{'.'.join(path + [key])}]\"] found in simulation observation but not in real observation"
                        )
                    check_observation_match(
                        sim_obs[key], real_obs[key], path=path + [key]
                    )
            else:
                assert (
                    sim_obs.shape == real_obs.shape
                ), f"Shape mismatch: obs[\"{'.'.join(path)}\"]: {sim_obs.shape} vs {real_obs.shape}"
                assert (
                    sim_obs.dtype == real_obs.dtype
                ), f"Dtype mismatch: obs[\"{'.'.join(path)}\"]: {sim_obs.dtype} vs {real_obs.dtype}"

        # Call the recursive function to check observations
        check_observation_match(sample_sim_obs, sample_real_obs)

    def close(self):
        self.agent.stop()

    def preprocess_sensor_data(
        self, sensor_data: Dict, sensor_names: Optional[List[str]] = None
    ):
        import cv2

        if sensor_names is None:
            sensor_names = list(sensor_data.keys())
        for sensor_name in sensor_names:
            sim_sensor_cfg = self.base_sim_env._sensor_configs[sensor_name]
            assert isinstance(sim_sensor_cfg, CameraConfig)
            target_h, target_w = sim_sensor_cfg.height, sim_sensor_cfg.width
            real_sensor_data = sensor_data[sensor_name]

            # crop to same aspect ratio
            for key in ["rgb", "depth"]:
                if key in real_sensor_data:
                    img = real_sensor_data[key][0].numpy()
                    xy_res = img.shape[:2]
                    crop_res = np.min(xy_res)
                    cutoff = (np.max(xy_res) - crop_res) // 2
                    if xy_res[0] == xy_res[1]:
                        pass
                    elif np.argmax(xy_res) == 0:
                        img = img[cutoff:-cutoff, :, :]
                    else:
                        img = img[:, cutoff:-cutoff, :]
                    real_sensor_data[key] = common.to_tensor(
                        cv2.resize(img, (target_w, target_h))
                    ).unsqueeze(0)

            sensor_data[sensor_name] = real_sensor_data
        return sensor_data

    def __getattr__(self, name):
        return getattr(self.base_sim_env, name)
