"""base environment for real robots, uses controller from reference simulation env"""
import copy
import time
from functools import cached_property
from typing import Dict, Union

import gymnasium as gym
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium.vector.utils import batch_space
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.utils import busy_wait

# lerobot imports
from lerobot.common.utils.utils import init_hydra_config
from tqdm import tqdm

# ManiSkill imports
from examples.baselines.ppo.ppo_rgb import Agent
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.utils import common, gym_utils, io_utils
from mani_skill.utils.visualization.misc import tile_images


# convert sim controller to real controller, only requires changing a few functions
class RealController:
    def __init__(self, controller, real_env):
        self.controller_instance = controller
        self.real_env_instance = real_env

    @property
    def qpos(self):
        return self.real_env_instance.robot_qpos

    @property
    def qvel(self):
        raise NotImplementedError()

    @property
    def device(self):
        return torch.device("cpu")

    def __getattr__(self, name):
        return getattr(self.controller_instance, name)


# control frequency automatically scheduled based on sim_env_id
# robot controller is used from
class BaseRealEnv(gym.Env):
    """
    # once the physical robot and configs are set up, allows easy testing of evaluation envirnoment
    # create image transform for raw camera outputs as necessary:
    def image_trans(img):
        return cv2.resize(img, (128,128))
    real_env = gym.make("RealGrabCube-v1", yaml_path="koch.yaml", keyframe_id="rest", control_mode="pd_joint_delta_pos", image_trans=image_trans)
    real_env = FlattenRGBDObservationWrapper(real_env, rgb=True, depth=False, state=True)

    # BaseRealEnv use pseudocode
    # stepping timing already configured, no need to manually sleep during evaluation:
    obs, info = real_env.reset() # resets robot to provided keyframe
    for _ in range(num_steps):
        action = policy.get_action(obs["rgb"], obs["state"])
        obs, _, _, _, _ = real_env.step(action) # action to raw control done by BaseRealEnv, using sim_env controller

    # Warning: safety of robot motors requires the following:
    # Correct calibration and setup of the robot
    # ManiSkill Controller limiting per step change in qpos
    # Qpos limits correctly defined in the URDF
    """

    sim_env_id = ""
    """simulation reference environment"""

    def __init__(self, yaml_path, keyframe_id, control_mode, image_trans=lambda x: x):
        self.keyframe_id = keyframe_id
        self.control_mode = control_mode
        self.image_trans = image_trans

        # TODO(xhin): generalize robot building to future robots
        # build real robot from yaml
        self.robot = hydra.utils.instantiate(init_hydra_config(yaml_path))
        self.robot.connect()

        # leader doesn't need torque enabled for policy eval on follower
        # if len(self.robot.leader_arms > 0):
        if isinstance(self.robot, ManipulatorRobot):
            for name in self.robot.leader_arms:
                self.robot.leader_arms[name].write(
                    "Torque_Enable", TorqueMode.DISABLED.value
                )
                print(f"MS3: Disabled {name} Torque")
                assert (
                    self.robot.leader_arms[name].read("Torque_Enable")
                    == TorqueMode.DISABLED.value
                ).all()

        # build reference environment
        self.sim_env = gym.make(self.sim_env_id, num_envs=1, control_mode=control_mode)

        # make real controller, wrapper of reference sim env controller
        self.controller = RealController(
            self.sim_env.agent.controllers[control_mode], self
        )

        # extract sim_env info
        self.sim_feq = self.sim_env.sim_freq
        self.control_freq = self.sim_env.control_freq
        self.control_low = torch.tensor(self.controller.config.lower)
        self.control_high = torch.tensor(self.controller.config.upper)
        self.single_action_space = self.sim_env.single_action_space
        self._orig_single_action_space = copy.deepcopy(self.sim_env.single_action_space)

        # TODO (xhin): controller interpolate sets drive targets but doesn't change self._target_qpos at each step per control step
        # TODO (xhin): also, it doesn't currently work with use_target at all
        if self.controller.config.interpolate:
            raise NotImplementedError(
                "Controller with interpolation not impemented for real robot yet"
            )

        self.robot_keyframe_qpos = torch.from_numpy(
            self.sim_env.agent.keyframes[keyframe_id].qpos
        )
        self.elapsed_steps = 0

        # hardcode the device to be cpu as we're running real environment
        self.device = torch.device("cpu")
        self.num_envs = 1

        # set necessary gym attributes
        obs = self.get_obs()
        self._init_raw_obs = common.to_cpu_tensor(obs)
        # initialize the cached properties
        self.single_observation_space
        self.observation_space

        qpos_limits = self.controller.articulation.get_qlimits()[
            0, self.controller.active_joint_indices
        ].cpu()
        self.qpos_lower_limits = qpos_limits[:, 0]
        self.qpos_upper_limits = qpos_limits[:, 1]

    @property
    def robot_qpos(self):
        if isinstance(self.robot, ManipulatorRobot):
            return torch.deg2rad(
                torch.tensor(self.robot.follower_arms["main"].read("Present_Position"))
            )
        else:
            raise NotImplementedError(
                "Support for this robot class is not implemented yet"
            )

    def apply_robot_action(self, action):
        if isinstance(self.robot, ManipulatorRobot):
            self.robot.send_action(torch.rad2deg(action))
        else:
            raise NotImplementedError(
                "Support for this robot class is not implemented yet"
            )

    def reset(self, seed, options):
        # this function moves the robot to the rest state, regardless of controller, uses pd_target_delta_qpos controller
        print(f"Moving to {self.keyframe_id} keyframe")
        if self.control_mode in [
            "pd_joint_pos",
            "pd_joint_delta_pos",
            "pd_joint_target_delta_pos",
        ]:
            max_rad_per_step = 0.01
            target_pos = self.robot_qpos
            # print(target_pos)
            # print(self.controller._get_joint_limits()[:, 0], type(self.controller._get_joint_limits()[:, 0]))
            # assert False
            for _ in tqdm(range(int(3 * self.control_freq))):
                start_loop_t = time.perf_counter()
                delta_step = (self.robot_keyframe_qpos - target_pos).clip(
                    min=-max_rad_per_step, max=max_rad_per_step
                )
                target_pos += delta_step
                # we can send the dist to start directly in, since we clip the actions
                self.apply_robot_action(target_pos)
                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / self.control_freq - dt_s)
        else:
            raise NotImplementedError(
                "Real robot reset for ee and joint_velocity controllers not implemented yet"
            )

        self.controller.reset()
        obs = self.get_obs()
        self.elapsed_steps = 0
        return obs, dict(reconfigure=False)

    def _get_obs_agent(self):
        return dict(qpos=common.batch(self.robot_qpos))

    def _get_obs_extra(self):
        return dict()

    # TODO (xhin):
    def _get_obs_sensor_data(self):
        sensor_obs = dict()
        if isinstance(self.robot, ManipulatorRobot):
            for name in self.robot.cameras:
                img = torch.tensor(self.image_trans(self.robot.cameras[name].read()))
                sensor_obs[name] = dict(rgb=img)
        else:
            raise NotImplementedError()
        return common.batch(sensor_obs)

    def _get_obs_with_sensor_data(self):
        return dict(
            agent=self._get_obs_agent(),
            extra=self._get_obs_extra(),
            sensor_param=dict(),
            sensor_data=self._get_obs_sensor_data(),
        )

    def get_obs(self):
        return self._get_obs_with_sensor_data()

    # all controllers work by solving for target qpos, sending _target_qpos to articulation in sim
    # if phyiscal robot moves via qpos control, this should work regardless of controller used
    def action_to_target_qpos(self, action):
        self.controller.set_action(action)
        return self.controller._target_qpos.clone()

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
        elif isinstance(action, dict):
            if "control_mode" in action:
                if action["control_mode"] != self.control_mode:
                    self.sim_env.agent.set_control_mode(action["control_mode"])
                    self.agent.controller.reset()
                    self.controller = RealController(
                        self.sim_env.agent.controllers[action["control_mode"]], self
                    )
                    self.controller.reset()
                action = common.to_tensor(action["action"], device=self.device)
                if action.shape == self._orig_single_action_space.shape:
                    action_is_unbatched = True
            else:
                assert isinstance(
                    self.agent, MultiAgent
                ), "Received a dictionary for an action but there are not multiple robots in the environment"
                # assume this is a multi-agent action
                action = common.to_tensor(action, device=self.device)
                for k, a in action.items():
                    if a.shape == self._orig_single_action_space[k].shape:
                        action_is_unbatched = True
                        break
            set_action = True
        else:
            raise TypeError(type(action))

        if self.elapsed_steps == 0:
            self.episode_start_time = time.perf_counter()
        else:
            control_step_time = (
                self.episode_start_time + (1 / self.control_freq) * self.elapsed_steps
            )
            curr_time = time.perf_counter()
            if curr_time > control_step_time:
                print(
                    f"Warning: control computation took too long, control step late by {curr_time - control_step_time} seconds. Consider reducing the env control_frequency and retrain policy"
                )
            else:
                busy_wait(control_step_time - curr_time)

        # TODO (xhin): iterate over num steps per control at sim_freq to correctly simulate control
        if set_action:
            if action_is_unbatched:
                action = common.batch(action)
            assert (
                action.shape[0] == 1
            ), f"real robot cannot process batched input, got {action.shape}"
            target_qpos = self.action_to_target_qpos(action)
            target_qpos = target_qpos.clip(
                min=self.qpos_lower_limits, max=self.qpos_upper_limits
            )
            self.apply_robot_action(target_qpos[0])

    def step(self, action, timer=[0]):
        # maybe even use the timer to warn if variance in time between steps
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

    # TODO (xhin): allow for different image trans for rendering
    def render(self):
        images = []
        if isinstance(self.robot, ManipulatorRobot):
            for name in self.robot.cameras:
                images.append(
                    torch.tensor(self.image_trans(self.robot.cameras[name].read()))
                )
        else:
            raise NotImplementedError()
        return tile_images(images)

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
