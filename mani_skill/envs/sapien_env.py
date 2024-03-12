import copy
import gc
import os
from collections import OrderedDict
from functools import cached_property
from typing import Any, Dict, List, Sequence, Tuple, Union

import dacite
import gymnasium as gym
import numpy as np
import sapien
import sapien.physx
import sapien.physx as physx
import sapien.render
import sapien.utils.viewer.control_window
import torch
from gymnasium.vector.utils import batch_space
from sapien.utils import Viewer

from mani_skill import logger
from mani_skill.agents import REGISTERED_AGENTS
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.envs.utils.observations import (
    sensor_data_to_pointcloud,
    sensor_data_to_rgbd,
)
from mani_skill.sensors.base_sensor import BaseSensor, BaseSensorConfig
from mani_skill.sensors.camera import (
    Camera,
    CameraConfig,
    parse_camera_cfgs,
    update_camera_cfgs_from_dict,
)
from mani_skill.sensors.depth_camera import StereoDepthCamera, StereoDepthCameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.common import (
    convert_observation_to_space,
    dict_merge,
    flatten_state_dict,
)
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.types import Array, SimConfig
from mani_skill.utils.visualization.misc import observations_to_images, tile_images


class BaseEnv(gym.Env):
    """Superclass for ManiSkill environments.

    Args:
        num_envs: number of parallel environments to run. By default this is 1, which means a CPU simulation is used. If greater than 1,
            then we initialize the GPU simulation setup. Note that not all environments are faster when simulated on the GPU due to limitations of
            GPU simulations. For example, environments with many moving objects are better simulated by parallelizing across CPUs.

        gpu_sim_backend: The GPU simulation backend to use (only used if the given num_envs argument is > 1). This affects the type of tensor
            returned by the environment for e.g. observations and rewards. Can be "torch" or "jax". Default is "torch"

        obs_mode: observation mode to be used. Must be one of ("state", "state_dict", "none", "sensor_data", "rgb", "rgbd", "pointcloud")

        reward_mode: reward mode to use. Must be one of ("normalized_dense", "dense", "sparse", "none"). With "none" the reward returned is always 0

        control_mode: control mode of the agent.
            "*" represents all registered controllers, and the action space will be a dict.

        render_mode: render mode registered in @SUPPORTED_RENDER_MODES.

        shader_dir (str): shader directory. Defaults to "default".
            "default", "rt", "rt-fast" are built-in options with SAPIEN. Other options are user-defined. "rt" means ray-tracing which results
            in more photorealistic renders but is slow, "rt-fast" is a lower quality but faster version of "rt".

        enable_shadow (bool): whether to enable shadow for lights. Defaults to False.

        sensor_cfgs (dict): configurations of sensors. See notes for more details.

        human_render_camera_cfgs (dict): configurations of human rendering cameras. Similar usage as @sensor_cfgs.

        robot_uids (Union[str, BaseAgent, List[Union[str, BaseAgent]]]): List of robots to instantiate and control in the environment.

        sim_cfg (Union[SimConfig, dict]): Configurations for simulation if used that override the environment defaults. If given
            a dictionary, it can just override specific attributes e.g. `sim_cfg=dict(scene_cfg=dict(solver_iterations=25))`. If
            passing in a SimConfig object, while typed, will override every attribute including the task defaults. Some environments
            define their own recommended default sim configurations via the `self._default_sim_cfg` attribute that generally should not be
            completely overriden. For a full detail/explanation of what is in the sim config see the type hints / go to the source
            https://github.com/haosulab/ManiSkill2/blob/main/mani_skill/utils/structs/types.py

        reconfiguration_freq (int): How frequently to call reconfigure when environment is reset via `self.reset(...)`
            Generally for most users who are not building tasks this does not need to be changed. The default is 0, which means
            the environment reconfigures upon creation, and never again.

        force_use_gpu_sim (bool): By default this is False. If the num_envs == 1, we use GPU sim if force_use_gpu_sim is True, otherwise we use CPU sim.

    Note:
        `sensor_cfgs` is used to update environement-specific sensor configurations.
        If the key is one of sensor names (e.g. a camera), the value will be applied to the corresponding sensor.
        Otherwise, the value will be applied to all sensors (but overridden by sensor-specific values).
    """

    # fmt: off
    SUPPORTED_ROBOTS: List[Union[str, Tuple[str]]] = None
    """Override this to enforce which robots or tuples of robots together are supported in the task. During env creation,
    setting robot_uids auto loads all desired robots into the scene, but not all tasks are designed to support some robot setups"""
    SUPPORTED_OBS_MODES = ("state", "state_dict", "none", "sensor_data", "rgb", "rgbd", "pointcloud")
    SUPPORTED_REWARD_MODES = ("normalized_dense", "dense", "sparse")
    SUPPORTED_RENDER_MODES = ("human", "rgb_array", "sensors")
    """The supported render modes. Human opens up a GUI viewer. rgb_array returns an rgb array showing the current environment state.
    sensors returns an rgb array but only showing all data collected by sensors as images put together"""

    metadata = {"render_modes": SUPPORTED_RENDER_MODES}

    physx_system: Union[sapien.physx.PhysxCpuSystem, sapien.physx.PhysxGpuSystem] = None

    _scene: ManiSkillScene = None
    """the main scene, which manages all sub scenes. In CPU simulation there is only one sub-scene"""

    agent: BaseAgent

    _sensors: Dict[str, BaseSensor]
    """all sensors configured in this environment"""
    _all_sensor_configs_parsed: Dict[str, BaseSensorConfig]
    """all sensor configurations parsed from self._sensor_configs and agent._sensor_configs"""
    _agent_sensor_configs_parsed: Dict[str, BaseSensorConfig]
    """all agent sensor configs parsed from agent._sensor_configs"""
    _human_render_cameras: Dict[str, Camera]
    """cameras used for rendering the current environment retrievable via `env.render_rgb_array()`. These are not used to generate observations"""
    _human_render_camera_configs: Dict[str, CameraConfig]
    """all camera configurations for cameras used for human render"""
    _human_render_camera_configs_parsed: Dict[str, CameraConfig]
    """all camera configurations parsed from self._human_render_camera_configs"""

    _hidden_objects: List[Union[Actor, Articulation]] = []
    """list of objects that are hidden during rendering when generating visual observations / running render_cameras()"""

    _main_rng: np.random.RandomState = None
    """main rng generator that generates episode seed sequences. For internal use only"""

    _episode_rng: np.random.RandomState = None
    """the numpy RNG that you can use to generate random numpy data"""

    def __init__(
        self,
        num_envs: int = 1,
        obs_mode: str = None,
        reward_mode: str = None,
        control_mode: str = None,
        render_mode: str = None,
        shader_dir: str = "default",
        enable_shadow: bool = False,
        sensor_cfgs: dict = None,
        human_render_camera_cfgs: dict = None,
        robot_uids: Union[str, BaseAgent, List[Union[str, BaseAgent]]] = None,
        sim_cfg: Union[SimConfig, dict] = dict(),
        reconfiguration_freq: int = 0,
        force_use_gpu_sim: bool = False,
    ):
        self.num_envs = num_envs
        self.reconfiguration_freq = reconfiguration_freq
        self._reconfig_counter = 0
        self._custom_sensor_configs = sensor_cfgs
        self._custom_human_render_camera_configs = human_render_camera_cfgs
        self.robot_uids = robot_uids
        if self.SUPPORTED_ROBOTS is not None:
            assert robot_uids in self.SUPPORTED_ROBOTS
        if num_envs > 1 or force_use_gpu_sim:
            if not sapien.physx.is_gpu_enabled():
                sapien.physx.enable_gpu()
            self.device = torch.device(
                "cuda"
            )  # TODO (stao): fix this for multi process support
        else:
            self.device = torch.device("cpu")

        # TODO (stao): move the merge code / handling union typed arguments outside here so classes inheriting BaseEnv only get
        # the already parsed sim config argument
        if isinstance(sim_cfg, SimConfig):
            sim_cfg = sim_cfg.dict()
        merged_gpu_sim_cfg = self._default_sim_cfg.dict()
        dict_merge(merged_gpu_sim_cfg, sim_cfg)
        self.sim_cfg = dacite.from_dict(data_class=SimConfig, data=merged_gpu_sim_cfg, config=dacite.Config(strict=True))
        """the final sim config after merging user overrides with the environment default"""
        sapien.physx.set_gpu_memory_config(**self.sim_cfg.gpu_memory_cfg.dict())
        self.shader_dir = shader_dir
        if self.shader_dir == "default":
            sapien.render.set_camera_shader_dir("minimal")
            sapien.render.set_picture_format("Color", "r8g8b8a8unorm")
            sapien.render.set_picture_format("ColorRaw", "r8g8b8a8unorm")
            sapien.render.set_picture_format("PositionSegmentation", "r16g16b16a16sint")
        elif self.shader_dir == "rt":
            sapien.render.set_camera_shader_dir("rt")
            sapien.render.set_viewer_shader_dir("rt")
            sapien.render.set_ray_tracing_samples_per_pixel(32)
            sapien.render.set_ray_tracing_path_depth(16)
            sapien.render.set_ray_tracing_denoiser(
                "optix"
            )  # TODO "optix or oidn?" previous value was just True
        elif self.shader_dir == "rt-fast":
            sapien.render.set_camera_shader_dir("rt")
            sapien.render.set_viewer_shader_dir("rt")
            sapien.render.set_ray_tracing_samples_per_pixel(2)
            sapien.render.set_ray_tracing_path_depth(1)
            sapien.render.set_ray_tracing_denoiser("optix")
        elif self.shader_dir == "rt-med":
            sapien.render.set_camera_shader_dir("rt")
            sapien.render.set_viewer_shader_dir("rt")
            sapien.render.set_ray_tracing_samples_per_pixel(4)
            sapien.render.set_ray_tracing_path_depth(3)
            sapien.render.set_ray_tracing_denoiser("optix")
        sapien.render.set_log_level(os.getenv("MS2_RENDERER_LOG_LEVEL", "warn"))

        # Set simulation and control frequency
        self._sim_freq = self.sim_cfg.sim_freq
        self._control_freq = self.sim_cfg.control_freq
        if self._sim_freq % self._control_freq != 0:
            logger.warn(
                f"sim_freq({self._sim_freq}) is not divisible by control_freq({self._control_freq}).",
            )
        self._sim_steps_per_control = self._sim_freq // self._control_freq

        # Observation mode
        if obs_mode is None:
            obs_mode = self.SUPPORTED_OBS_MODES[0]
        if obs_mode not in self.SUPPORTED_OBS_MODES:
            raise NotImplementedError("Unsupported obs mode: {}".format(obs_mode))
        self._obs_mode = obs_mode

        # Reward mode
        if reward_mode is None:
            reward_mode = self.SUPPORTED_REWARD_MODES[0]
        if reward_mode not in self.SUPPORTED_REWARD_MODES:
            raise NotImplementedError("Unsupported reward mode: {}".format(reward_mode))
        self._reward_mode = reward_mode

        # Control mode
        self._control_mode = control_mode
        # TODO(jigu): Support dict action space
        if control_mode == "*":
            raise NotImplementedError("Multiple controllers are not supported yet.")

        # Render mode
        self.render_mode = render_mode
        self._viewer = None

        # Lighting
        self.enable_shadow = enable_shadow

        # Use a fixed (main) seed to enhance determinism
        self._main_seed = None
        self._set_main_rng(2022)
        self._elapsed_steps = (
            torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
            if physx.is_gpu_enabled()
            else 0
        )
        obs, _ = self.reset(seed=2022, options=dict(reconfigure=True))
        if physx.is_gpu_enabled():
            obs = sapien_utils.to_numpy(obs)
        self._init_raw_obs = obs.copy()
        """the raw observation returned by the env.reset. Useful for future observation wrappers to use to auto generate observation spaces"""
        self._init_raw_state = sapien_utils.to_numpy(self.get_state_dict())
        """the initial raw state returned by env.get_state. Useful for reconstructing state dictionaries from flattened state vectors"""

        self.action_space = self.agent.action_space
        self.single_action_space = self.agent.single_action_space
        self._orig_single_action_space = copy.deepcopy(self.single_action_space)
        # initialize the cached properties
        self.single_observation_space
        self.observation_space

    def _update_obs_space(self, obs: Any):
        """call this function if you modify the observations returned by env.step and env.reset via an observation wrapper. The given observation must be a numpy array"""
        self._init_raw_obs = obs
        del self.single_observation_space
        del self.observation_space
        self.single_observation_space
        self.observation_space

    @cached_property
    def single_observation_space(self):
        if self.num_envs > 1:
            return convert_observation_to_space(self._init_raw_obs, unbatched=True)
        else:
            return convert_observation_to_space(self._init_raw_obs)

    @cached_property
    def observation_space(self):
        if self.num_envs > 1:
            return batch_space(self.single_observation_space, n=self.num_envs)
        else:
            return self.single_observation_space

    @property
    def _default_sim_cfg(self):
        return SimConfig()
    def _load_agent(self):
        agents = []
        robot_uids = self.robot_uids
        if robot_uids is not None:
            if not isinstance(robot_uids, tuple):
                robot_uids = [robot_uids]
            for i, robot_uid in enumerate(robot_uids):
                if isinstance(robot_uid, type(BaseAgent)):
                    agent_cls = robot_uid
                else:
                    if robot_uid not in REGISTERED_AGENTS:
                        raise RuntimeError(
                            f"Agent {robot_uid} not found in the dict of registered agents. If the id is not a typo then make sure to apply the @register_agent() decorator."
                        )
                    agent_cls = REGISTERED_AGENTS[robot_uid].agent_cls
                agent: BaseAgent = agent_cls(
                    self._scene,
                    self._control_freq,
                    self._control_mode,
                    agent_idx=i if len(robot_uids) > 1 else None,
                )
                agents.append(agent)
        if len(agents) == 1:
            self.agent = agents[0]
        else:
            self.agent = MultiAgent(agents)
        # TODO (stao): do we stil need this?
        # set_articulation_render_material(self.agent.robot, specular=0.9, roughness=0.3)
    @property
    def _sensor_configs(
        self,
    ) -> Union[
        BaseSensorConfig, Sequence[BaseSensorConfig], Dict[str, BaseSensorConfig]
    ]:
        """Add (non-agent) sensors to the environment by returning sensor configurations"""
        return []
    @property
    def _human_render_camera_configs(
        self,
    ) -> Union[
        BaseSensorConfig, Sequence[BaseSensorConfig], Dict[str, BaseSensorConfig]
    ]:
        """Add cameras for rendering when using render_mode='rgb_array' """
        return []

    @property
    def sim_freq(self):
        return self._sim_freq

    @property
    def control_freq(self):
        return self._control_freq

    @property
    def sim_timestep(self):
        return 1.0 / self._sim_freq

    @property
    def control_timestep(self):
        return 1.0 / self._control_freq

    @property
    def control_mode(self):
        return self.agent.control_mode

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    # ---------------------------------------------------------------------------- #
    # Observation
    # ---------------------------------------------------------------------------- #
    @property
    def obs_mode(self):
        return self._obs_mode

    def get_obs(self, info: Dict = None):
        """
        Return the current observation of the environment. User may call this directly to get the current observation
        as opposed to taking a step with actions in the environment.

        Note that some tasks use info of the current environment state to populate the observations to avoid having to
        compute slow operations twice. For example a state based observation may wish to include a boolean indicating
        if a robot is grasping an object. Computing this boolean correctly is slow, so it is preferable to generate that
        data in the info object by overriding the `self.evaluate` function.

        Args:
            info (Dict): The info object of the environment. Generally should always be the result of `self.get_info()`.
                If this is None (the default), this function will call `self.get_info()` itself
        """
        if info is None:
            info = self.get_info()
        if self._obs_mode == "none":
            # Some cases do not need observations, e.g., MPC
            return OrderedDict()
        elif self._obs_mode == "state":
            state_dict = self._get_obs_state_dict(info)
            obs = flatten_state_dict(state_dict, use_torch=True, device=self.device)
        elif self._obs_mode == "state_dict":
            obs = self._get_obs_state_dict(info)
        elif self._obs_mode in ["sensor_data", "rgbd", "rgb", "pointcloud"]:
            obs = self._get_obs_with_sensor_data(info)
            if self._obs_mode == "rgbd":
                obs = sensor_data_to_rgbd(obs, self._sensors, rgb=True, depth=True)
            elif self._obs_mode == "rgb":
                # TODO (stao): we can optmize this by not taking the PositionSegmentation texture at all.
                obs = sensor_data_to_rgbd(obs, self._sensors, rgb=True, depth=False)
            elif self.obs_mode == "pointcloud":
                obs = sensor_data_to_pointcloud(obs, self._sensors)
        else:
            raise NotImplementedError(self._obs_mode)
        return obs

    def _get_obs_state_dict(self, info: Dict):
        """Get (ground-truth) state-based observations."""
        return OrderedDict(
            agent=self._get_obs_agent(),
            extra=self._get_obs_extra(info),
        )

    def _get_obs_agent(self):
        """Get observations from the agent's sensors, e.g., proprioceptive sensors."""
        return self.agent.get_proprioception()

    def _get_obs_extra(self, info: Dict):
        """Get task-relevant extra observations."""
        return OrderedDict()

    def capture_sensor_data(self):
        """Capture data from all sensors (non-blocking)"""
        for sensor in self._sensors.values():
            sensor.capture()

    def get_sensor_obs(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get raw sensor data for use as observations."""
        sensor_data = OrderedDict()
        for name, sensor in self._sensors.items():
            sensor_data[name] = sensor.get_obs()
        return sensor_data

    def get_sensor_images(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get raw sensor data as images for visualization purposes."""
        sensor_data = OrderedDict()
        for name, sensor in self._sensors.items():
            sensor_data[name] = sensor.get_images()
        return sensor_data

    def get_sensor_params(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get all sensor parameters."""
        params = OrderedDict()
        for name, sensor in self._sensors.items():
            params[name] = sensor.get_params()
        return params

    def _get_obs_with_sensor_data(self, info: Dict) -> OrderedDict:
        for obj in self._hidden_objects:
            obj.hide_visual()
        self._scene.update_render()
        self.capture_sensor_data()
        return OrderedDict(
            agent=self._get_obs_agent(),
            extra=self._get_obs_extra(info),
            sensor_param=self.get_sensor_params(),
            sensor_data=self.get_sensor_obs(),
        )

    @property
    def robot_link_ids(self):
        """Get link ids for the robot. This is used for segmentation observations."""
        return self.agent.robot_link_ids

    # -------------------------------------------------------------------------- #
    # Reward mode
    # -------------------------------------------------------------------------- #
    @property
    def reward_mode(self):
        return self._reward_mode

    def get_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        if self._reward_mode == "sparse":
            if "success" in info:
                if "fail" in info:
                    if isinstance(info["success"], torch.Tensor):
                        reward = info["success"].to(torch.float) - info["fail"].to(torch.float)
                    else:
                        reward = info["success"] - info["fail"]
                else:
                    reward = info["success"]
            else:
                if "fail" in info:
                    reward = -info["fail"]
                else:
                    reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        elif self._reward_mode == "dense":
            reward = self.compute_dense_reward(obs=obs, action=action, info=info)
        elif self._reward_mode == "normalized_dense":
            reward = self.compute_normalized_dense_reward(
                obs=obs, action=action, info=info
            )
        elif self._reward_mode == "none":
            reward = 0
            if self.num_envs > 1:
                reward = torch.zeros((self.num_envs, ), dtype=torch.float, device=self.device)
        else:
            raise NotImplementedError(self._reward_mode)
        return reward

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        raise NotImplementedError()

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        raise NotImplementedError()

    # -------------------------------------------------------------------------- #
    # Reconfigure
    # -------------------------------------------------------------------------- #
    def _reconfigure(self):
        """Reconfigure the simulation scene instance.
        This function clears the previous scene and creates a new one.

        Note this function is not always called when an environment is reset, and
        should only be used if any agents, assets, sensors, lighting need to change
        to save compute time.

        Tasks like PegInsertionSide and TurnFaucet will call this each time as the peg
        shape changes each time and the faucet model changes each time respectively.
        """

        self._clear()
        # load everything into the scene first before initializing anything
        self._setup_scene()
        self._load_agent()
        self._load_scene()

        self._load_lighting()

        if sapien.physx.is_gpu_enabled():
            self._scene._setup_gpu()
            self._scene._gpu_fetch_all()
        # for GPU sim, we have to setup sensors after we call setup gpu in order to enable loading mounted sensors as they depend on GPU buffer data
        self._setup_sensors()
        if self._viewer is not None:
            self._setup_viewer()
        self._reconfig_counter = self.reconfiguration_freq

    def _after_reconfigure(self):
        """Add code here that should run immediately after self._reconfigure() is called. The torch RNG context is still active so RNG is still
        seeded here by self._episode_seed. This is useful if you need to run something that only happens after reconfiguration but need the
        GPU initialized so that you can check e.g. collisons, poses etc."""

    def _load_scene(self):
        """Loads all objects like actors and articulations into the scene. Called by `self._reconfigure`"""

    # TODO (stao): refactor this into sensor API
    def _setup_sensors(self):
        """Setup sensor configurations and the sensor objects in the scene. Called by `self._reconfigure`"""

        # First create all the configurations
        self._all_sensor_configs_parsed = OrderedDict()

        # Add task/external sensors
        self._all_sensor_configs_parsed.update(parse_camera_cfgs(self._sensor_configs))

        # Add agent sensors
        self._agent_sensor_configs_parsed = OrderedDict()
        self._agent_sensor_configs_parsed = parse_camera_cfgs(self.agent._sensor_configs)
        self._all_sensor_configs_parsed.update(self._agent_sensor_configs_parsed)

        # Add human render camera configs
        self._human_render_camera_configs_parsed = parse_camera_cfgs(
            self._human_render_camera_configs
        )

        # Override camera configurations with user supplied configurations
        if self._custom_sensor_configs is not None:
            update_camera_cfgs_from_dict(
                self._all_sensor_configs_parsed, self._custom_sensor_configs
            )
        if self._custom_human_render_camera_configs is not None:
            update_camera_cfgs_from_dict(
                self._human_render_camera_configs_parsed,
                self._custom_human_render_camera_configs,
            )

        # Now we instantiate the actual sensor objects
        self._sensors = OrderedDict()

        for uid, sensor_cfg in self._all_sensor_configs_parsed.items():
            if uid in self._agent_sensor_configs_parsed:
                articulation = self.agent.robot
            else:
                articulation = None
            if isinstance(sensor_cfg, StereoDepthCameraConfig):
                sensor_cls = StereoDepthCamera
            elif isinstance(sensor_cfg, CameraConfig):
                sensor_cls = Camera
            self._sensors[uid] = sensor_cls(
                sensor_cfg,
                self._scene,
                articulation=articulation,
            )

        # Cameras for rendering only
        self._human_render_cameras = OrderedDict()
        for uid, camera_cfg in self._human_render_camera_configs_parsed.items():
            self._human_render_cameras[uid] = Camera(
                camera_cfg,
                self._scene,
            )

        self._scene.sensors = self._sensors
        self._scene.human_render_cameras = self._human_render_cameras

    def _load_lighting(self):
        """Loads lighting into the scene. Called by `self._reconfigure`. If not overriden will set some simple default lighting"""

        shadow = self.enable_shadow
        self._scene.set_ambient_light([0.3, 0.3, 0.3])
        # Only the first of directional lights can have shadow
        self._scene.add_directional_light(
            [1, 1, -1], [1, 1, 1], shadow=shadow, shadow_scale=5, shadow_map_size=2048
        )
        self._scene.add_directional_light([0, 0, -1], [1, 1, 1])

    # -------------------------------------------------------------------------- #
    # Reset
    # -------------------------------------------------------------------------- #
    def reset(self, seed=None, options=None):
        """
        Reset the ManiSkill environment. If options["env_idx"] is given, will only reset the selected parallel environments. If
        options["reconfigure"] is True, will call self._reconfigure() which deletes the entire physx scene and reconstructs everything.
        Users building custom tasks generally do not need to override this function.

        Note that ManiSkill always holds two RNG states, a main RNG, and an episode RNG. The main RNG is used purely to sample an episode seed which
        helps with reproducibility of episodes and is for internal use only. The episode RNG is used by the environment/task itself to
        e.g. randomize object positions, randomize assets etc. Episode RNG is accessible by using torch.rand (recommended) which is seeded with a
        RNG context or the numpy alternative via `self._episode_rng`

        Upon environment creation via gym.make, the main RNG is set with a fixed seed of 2022.
        During each reset call, if seed is None, main RNG is unchanged and an episode seed is sampled from the main RNG to create the episode RNG.
        If seed is not None, main RNG is set to that seed and the episode seed is also set to that seed. This design means the main RNG determines
        the episode RNG deterministically.

        """
        if options is None:
            options = dict()

        self._set_main_rng(seed)
        # we first set the first episode seed to allow environments to use it to reconfigure the environment with a seed
        self._set_episode_rng(seed)

        reconfigure = options.get("reconfigure", False)
        reconfigure = reconfigure or (
            self._reconfig_counter == 0 and self.reconfiguration_freq != 0
        )
        if reconfigure:
            with torch.random.fork_rng():
                torch.manual_seed(seed=self._episode_seed)
                self._reconfigure()
                self._after_reconfigure()
        if "env_idx" in options:
            env_idx = options["env_idx"]
            self._scene._reset_mask = torch.zeros(
                self.num_envs, dtype=bool, device=self.device
            )
            self._scene._reset_mask[env_idx] = True
        else:
            env_idx = torch.arange(0, self.num_envs, device=self.device)
            self._scene._reset_mask = torch.ones(
                self.num_envs, dtype=bool, device=self.device
            )
        if physx.is_gpu_enabled():
            self._elapsed_steps[env_idx] = 0
        else:
            self._elapsed_steps = 0

        if not reconfigure:
            self._clear_sim_state()
        if self.reconfiguration_freq != 0:
            self._reconfig_counter -= 1
        # Set the episode rng again after reconfiguration to guarantee seed reproducibility
        self._set_episode_rng(self._episode_seed)
        self.agent.reset()
        with torch.random.fork_rng():
            torch.manual_seed(self._episode_seed)
            self._initialize_episode(env_idx)
        # reset the reset mask back to all ones so any internal code in maniskill can continue to manipulate all scenes at once as usual
        self._scene._reset_mask = torch.ones(
            self.num_envs, dtype=bool, device=self.device
        )
        if physx.is_gpu_enabled():
            # ensure all updates to object poses and configurations are applied on GPU after task initialization
            self._scene._gpu_apply_all()
            self._scene.px.gpu_update_articulation_kinematics()
            self._scene._gpu_fetch_all()
        obs = self.get_obs()
        if not physx.is_gpu_enabled():
            obs = sapien_utils.to_numpy(sapien_utils.unbatch(obs))
            self._elapsed_steps = 0
        return obs, {}

    def _set_main_rng(self, seed):
        """Set the main random generator which is only used to set the seed of the episode RNG to improve reproducibility.

        Note that while _set_main_rng and _set_episode_rng are setting a seed and numpy random state, when using GPU sim
        parallelization it is highly recommended to use torch random functions as they will make things run faster. The use
        of torch random functions when building tasks in ManiSkill are automatically seeded via `torch.random.fork`
        """
        if seed is None:
            if self._main_seed is not None:
                return
            seed = np.random.RandomState().randint(2**32)
        self._main_seed = seed
        self._main_rng = np.random.RandomState(self._main_seed)

    def _set_episode_rng(self, seed):
        """Set the random generator for current episode."""
        if seed is None:
            self._episode_seed = self._main_rng.randint(2**32)
        else:
            self._episode_seed = seed
        self._episode_rng = np.random.RandomState(self._episode_seed)

    def _initialize_episode(self, env_idx: torch.Tensor):
        """Initialize the episode, e.g., poses of actors and articulations, as well as task relevant data like randomizing
        goal positions
        """

    def _clear_sim_state(self):
        """Clear simulation state (velocities)"""
        for actor in self._scene.actors.values():
            if actor.px_body_type == "static":
                continue
            actor.set_linear_velocity([0, 0, 0])
            actor.set_angular_velocity([0, 0, 0])
        for articulation in self._scene.articulations.values():
            articulation.set_qvel(np.zeros(articulation.max_dof))
            articulation.set_root_linear_velocity([0, 0, 0])
            articulation.set_root_angular_velocity([0, 0, 0])
        if physx.is_gpu_enabled():
            self._scene._gpu_apply_all()
            self._scene._gpu_fetch_all()
            # TODO (stao): This may be an unnecessary fetch and apply.

    # -------------------------------------------------------------------------- #
    # Step
    # -------------------------------------------------------------------------- #

    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):
        action = self.step_action(action)
        self._elapsed_steps += 1
        info = self.get_info()
        obs = self.get_obs(info)
        reward = self.get_reward(obs=obs, action=action, info=info)
        if "success" in info:

            if "fail" in info:
                terminated = torch.logical_or(info["success"], info["fail"])
            else:
                terminated = info["success"].clone()
        else:
            if "fail" in info:
                terminated = info["fail"].clone()
            else:
                terminated = torch.zeros(self.num_envs, dtype=bool, device=self.device)

        if physx.is_gpu_enabled():
            return (
                obs,
                reward,
                terminated,
                torch.zeros(self.num_envs, dtype=bool, device=self.device),
                info,
            )
        else:
            # On CPU sim mode, we always return numpy / python primitives without any batching.
            return sapien_utils.unbatch(
                sapien_utils.to_numpy(obs),
                sapien_utils.to_numpy(reward),
                sapien_utils.to_numpy(terminated),
                False,
                sapien_utils.to_numpy(info),
            )

    def step_action(
        self, action: Union[None, np.ndarray, torch.Tensor, Dict]
    ) -> Union[None, torch.Tensor]:
        set_action = False
        action_is_unbatched = False
        if action is None:  # simulation without action
            pass
        elif isinstance(action, np.ndarray) or isinstance(action, torch.Tensor):
            action = sapien_utils.to_tensor(action)
            if action.shape == self._orig_single_action_space.shape:
                action_is_unbatched = True
            set_action = True
        elif isinstance(action, dict):
            if "control_mode" in action:
                if action["control_mode"] != self.agent.control_mode:
                    self.agent.set_control_mode(action["control_mode"])
                    self.agent.controller.reset()
                action = sapien_utils.to_tensor(action["action"])
            else:
                assert isinstance(
                    self.agent, MultiAgent
                ), "Received a dictionary for an action but there are not multiple robots in the environment"
                # assume this is a multi-agent action
                action = sapien_utils.to_tensor(action)
                for k, a in action.items():
                    if a.shape == self._orig_single_action_space[k].shape:
                        action_is_unbatched = True
                        break
            set_action = True
        else:
            raise TypeError(type(action))

        if set_action:
            if self.num_envs == 1 and action_is_unbatched:
                action = sapien_utils.batch(action)
            self.agent.set_action(action)
            if physx.is_gpu_enabled():
                self._scene.px.gpu_apply_articulation_target_position()
                self._scene.px.gpu_apply_articulation_target_velocity()
        self._before_control_step()
        for _ in range(self._sim_steps_per_control):
            self.agent.before_simulation_step()
            self._before_simulation_step()
            with sapien.profile("step_i"):
                self._scene.step()
            self._after_simulation_step()
        self._after_control_step()
        if physx.is_gpu_enabled():
            self._scene._gpu_fetch_all()
        return action

    def evaluate(self) -> dict:
        """
        Evaluate whether the environment is currently in a success state by returning a dictionary with a "success" key or
        a failure state via a "fail" key

        This function may also return additional data that has been computed (e.g. is the robot grasping some object) that may be
        reused when generating observations and rewards.
        """
        raise NotImplementedError

    def get_info(self):
        """
        Get info about the current environment state, include elapsed steps and evaluation information
        """
        info = dict(
            elapsed_steps=self.elapsed_steps
            if not physx.is_gpu_enabled()
            else self._elapsed_steps.clone()
        )
        info.update(self.evaluate())
        return info

    def _before_control_step(self):
        """Code that runs before each action has been taken.
        On GPU simulation this is called before observations are fetched from the GPU buffers."""
    def _after_control_step(self):
        """Code that runs after each action has been taken.
        On GPU simulation this is called right before observations are fetched from the GPU buffers."""

    def _before_simulation_step(self):
        """Code to run right before each physx_system.step is called"""
    def _after_simulation_step(self):
        """Code to run right after each physx_system.step is called"""

    # -------------------------------------------------------------------------- #
    # Simulation and other gym interfaces
    # -------------------------------------------------------------------------- #
    def _set_scene_config(self):
        physx.set_scene_config(**self.sim_cfg.scene_cfg.dict())
        physx.set_default_material(**self.sim_cfg.default_materials_cfg.dict())

    def _setup_scene(self):
        """Setup the simulation scene instance.
        The function should be called in reset(). Called by `self._reconfigure`"""
        self._set_scene_config()
        if sapien.physx.is_gpu_enabled():
            self.physx_system = sapien.physx.PhysxGpuSystem()
            # Create the scenes in a square grid
            sub_scenes = []
            scene_grid_length = int(np.ceil(np.sqrt(self.num_envs)))
            for scene_idx in range(self.num_envs):
                scene_x, scene_y = (
                    scene_idx % scene_grid_length,
                    scene_idx // scene_grid_length,
                )
                scene = sapien.Scene(
                    systems=[self.physx_system, sapien.render.RenderSystem()]
                )
                scene.physx_system.set_scene_offset(
                    scene,
                    [
                        scene_x * self.sim_cfg.spacing,
                        scene_y * self.sim_cfg.spacing,
                        0,
                    ],
                )
                sub_scenes.append(scene)
        else:
            self.physx_system = sapien.physx.PhysxCpuSystem()
            sub_scenes = [
                sapien.Scene([self.physx_system, sapien.render.RenderSystem()])
            ]
        # create a "global" scene object that users can work with that is linked with all other scenes created
        self._scene = ManiSkillScene(sub_scenes, sim_cfg=self.sim_cfg, device=self.device)
        self.physx_system.timestep = 1.0 / self._sim_freq

    def _clear(self):
        """Clear the simulation scene instance and other buffers.
        The function can be called in reset() before a new scene is created.
        Called by `self._reconfigure` and when the environment is closed/deleted
        """
        self._close_viewer()
        self.agent = None
        self._sensors = OrderedDict()
        self._human_render_cameras = OrderedDict()
        self._scene = None
        self._hidden_objects = []

    def close(self):
        self._clear()
        gc.collect()  # force gc to collect which releases most GPU memory

    def _close_viewer(self):
        if self._viewer is None:
            return
        self._viewer.close()
        self._viewer = None

    # -------------------------------------------------------------------------- #
    # Simulation state (required for MPC)
    # -------------------------------------------------------------------------- #
    def get_actors(self) -> List[sapien.Entity]:
        return self._scene.get_all_actors()

    def get_articulations(self) -> List[physx.PhysxArticulation]:
        articulations = self._scene.get_all_articulations()
        # NOTE(jigu): There might be dummy articulations used by controllers.
        # TODO(jigu): Remove dummy articulations if exist.
        return articulations

    def get_state_dict(self):
        """
        Get environment state dictionary. Override to include task information (e.g., goal)
        """
        return self._scene.get_sim_state()

    def get_state(self):
        """
        Get environment state as a flat vector, which is just a ordered flattened version of the state_dict.

        Users should not override this function
        """
        return flatten_state_dict(self.get_state_dict(), use_torch=True)

    def set_state_dict(self, state: Dict):
        """
        Set environment state with a state dictionary. Override to include task information (e.g., goal)

        Note that it is recommended to keep around state dictionaries as opposed to state vectors. With state vectors we assume
        the order of data in the vector is the same exact order that would be returned by flattening the state dictionary you get from
        `env.get_state_dict()` or the result of `env.get_state()`
        """
        self._scene.set_sim_state(state)
        if physx.is_gpu_enabled():
            self._scene._gpu_apply_all()
            self._scene.px.gpu_update_articulation_kinematics()
            self._scene._gpu_fetch_all()

    def set_state(self, state: Array):
        """
        Set environment state with a flat state vector. Internally this reconstructs the state dictionary and calls `env.set_state_dict`

        Users should not override this function
        """
        state_dict = dict()
        state_dict["actors"] = dict()
        state_dict["articulations"] = dict()
        KINEMATIC_DIM = 13  # [pos, quat, lin_vel, ang_vel]
        start = 0
        for actor_id in self._init_raw_state["actors"].keys():
            state_dict["actors"][actor_id] = state[:, start : start + KINEMATIC_DIM]
            start += KINEMATIC_DIM
        for art_id, art_state in self._init_raw_state["articulations"].items():
            size = art_state.shape[-1]
            state_dict["articulations"][art_id] = state[:, start : start + size]
            start += size
        self.set_state_dict(state_dict)

    # -------------------------------------------------------------------------- #
    # Visualization
    # -------------------------------------------------------------------------- #
    @property
    def viewer(self):
        return self._viewer

    def _setup_viewer(self):
        """Setup the interactive viewer.

        The function should be called after a new scene is configured.
        In subclasses, this function can be overridden to set viewer cameras.

        Called by `self._reconfigure`
        """
        # TODO (stao): handle GPU parallel sim rendering code:
        if physx.is_gpu_enabled():
            self._viewer_scene_idx = 0
        # CAUTION: `set_scene` should be called after assets are loaded.
        self._viewer.set_scene(self._scene.sub_scenes[0])
        control_window: sapien.utils.viewer.control_window.ControlWindow = (
            sapien_utils.get_obj_by_type(
                self._viewer.plugins, sapien.utils.viewer.control_window.ControlWindow
            )
        )
        control_window.show_joint_axes = False
        control_window.show_camera_linesets = False
        if "render_camera" in self._human_render_cameras:
            self._viewer.set_camera_pose(
                self._human_render_cameras["render_camera"].camera.global_pose[0].sp
            )

    def render_human(self):
        for obj in self._hidden_objects:
            obj.show_visual()
        if self._viewer is None:
            self._viewer = Viewer()
            self._setup_viewer()
        if physx.is_gpu_enabled() and self._scene._gpu_sim_initialized:
            self.physx_system.sync_poses_gpu_to_cpu()
        self._viewer.render()
        for obj in self._hidden_objects:
            obj.hide_visual()
        return self._viewer

    def render_rgb_array(self, camera_name: str = None):
        """Returns an RGB array / image of size (num_envs, H, W, 3) of the current state of the environment.
        This is captured by any of the registered human render cameras. If a camera_name is given, only data from that camera is returned.
        Otherwise all camera data is captured and returned as a single batched image"""
        for obj in self._hidden_objects:
            obj.show_visual()
        self._scene.update_render()
        images = []
        if physx.is_gpu_enabled():
            for name in self._scene.human_render_cameras.keys():
                camera_group = self._scene.camera_groups[name]
                if camera_name is not None and name != camera_name:
                    continue
                camera_group.take_picture()
                rgb = camera_group.get_picture_cuda("Color").torch()[..., :3].clone()
                images.append(rgb)
        else:
            for name, camera in self._scene.human_render_cameras.items():
                if camera_name is not None and name != camera_name:
                    continue
                camera.capture()
                if self.shader_dir == "default":
                    rgb = (camera.get_picture("Color")[..., :3]).to(torch.uint8)
                else:
                    rgb = (camera.get_picture("Color")[..., :3] * 255).to(torch.uint8)
                images.append(rgb)
        if len(images) == 0:
            return None
        if len(images) == 1:
            return images[0]
        for obj in self._hidden_objects:
            obj.hide_visual()
        return tile_images(images)

    def render_sensors(self):
        """
        Renders all sensors that the agent can use and see and displays them
        """
        for obj in self._hidden_objects:
            obj.hide_visual()
        images = []
        self._scene.update_render()
        self.capture_sensor_data()
        sensor_images = self.get_sensor_obs()
        for sensor_images in sensor_images.values():
            images.extend(observations_to_images(sensor_images))
        return tile_images(images)

    def render(self):
        """
        Either opens a viewer if render_mode is "human", or returns an array that you can use to save videos.

        render_mode is "rgb_array", usually a higher quality image is rendered for the purpose of viewing only.

        if render_mode is "sensors", all visual observations the agent can see is provided
        """
        if self.render_mode is None:
            raise RuntimeError("render_mode is not set.")
        if self.render_mode == "human":
            return self.render_human()
        elif self.render_mode == "rgb_array":
            return self.render_rgb_array()
        elif self.render_mode == "sensors":
            return self.render_sensors()
        else:
            raise NotImplementedError(f"Unsupported render mode {self.render_mode}.")

    # TODO (stao): re implement later
    # # ---------------------------------------------------------------------------- #
    # # Advanced
    # # ---------------------------------------------------------------------------- #
    # def gen_scene_pcd(self, num_points: int = int(1e5)) -> np.ndarray:
    #     """Generate scene point cloud for motion planning, excluding the robot"""
    #     meshes = []
    #     articulations = self._scene.get_all_articulations()
    #     if self.agent is not None:
    #         articulations.pop(articulations.index(self.agent.robot))
    #     for articulation in articulations:
    #         articulation_mesh = merge_meshes(get_articulation_meshes(articulation))
    #         if articulation_mesh:
    #             meshes.append(articulation_mesh)

    #     for actor in self._scene.get_all_actors():
    #         actor_mesh = merge_meshes(get_component_meshes(actor))
    #         if actor_mesh:
    #             meshes.append(
    #                 actor_mesh.apply_transform(
    #                     actor.get_pose().to_transformation_matrix()
    #                 )
    #             )

    #     scene_mesh = merge_meshes(meshes)
    #     scene_pcd = scene_mesh.sample(num_points)
    #     return scene_pcd
