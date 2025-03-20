import copy
import gc
import os
from functools import cached_property
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import dacite
import gymnasium as gym
import numpy as np
import sapien
import sapien.physx as physx
import sapien.render
import sapien.utils.viewer.control_window
import torch
from gymnasium.vector.utils import batch_space

from mani_skill import logger
from mani_skill.agents import REGISTERED_AGENTS
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.envs.utils.observations import (
    parse_obs_mode_to_struct,
    sensor_data_to_pointcloud,
)
from mani_skill.envs.utils.randomization.batched_rng import BatchedRNG
from mani_skill.envs.utils.system.backend import parse_sim_and_render_backend, CPU_SIM_BACKENDS
from mani_skill.sensors.base_sensor import BaseSensor, BaseSensorConfig
from mani_skill.sensors.camera import (
    Camera,
    CameraConfig,
    parse_camera_configs,
    update_camera_configs_from_dict,
)
from mani_skill.sensors.depth_camera import StereoDepthCamera, StereoDepthCameraConfig
from mani_skill.utils import common, gym_utils, sapien_utils
from mani_skill.utils.structs import Actor, Articulation
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array, SimConfig
from mani_skill.utils.visualization.misc import tile_images
from mani_skill.viewer import create_viewer


class BaseEnv(gym.Env):
    """Superclass for ManiSkill environments.

    Args:
        num_envs: number of parallel environments to run. By default this is 1, which means a CPU simulation is used. If greater than 1,
            then we initialize the GPU simulation setup. Note that not all environments are faster when simulated on the GPU due to limitations of
            GPU simulations. For example, environments with many moving objects are better simulated by parallelizing across CPUs.

        obs_mode: observation mode to be used. Must be one of ("state", "state_dict", "none", "sensor_data", "rgb", "depth", "segmentation", "rgbd", "rgb+depth", "rgb+depth+segmentation", "rgb+segmentation", "depth+segmentation", "pointcloud")
            The obs_mode is mostly for convenience to automatically optimize/setup all sensors/cameras for the given observation mode to render the correct data and try to ignore unecesary rendering.
            For the most advanced use cases (e.g. you have 1 RGB only camera and 1 depth only camera)

        reward_mode: reward mode to use. Must be one of ("normalized_dense", "dense", "sparse", "none"). With "none" the reward returned is always 0

        control_mode: control mode of the agent.
            "*" represents all registered controllers, and the action space will be a dict.

        render_mode: render mode registered in @SUPPORTED_RENDER_MODES.

        shader_dir (Optional[str]): shader directory. Defaults to None.
            Setting this will override the shader used for all cameras in the environment. This is legacy behavior kept for backwards compatibility.
            The proper way to change the shaders used for cameras is to either change the environment code or pass in sensor_configs/human_render_camera_configs with the desired shaders.


            Previously the options are "default", "rt", "rt-fast". "rt" means ray-tracing which results
            in more photorealistic renders but is slow, "rt-fast" is a lower quality but faster version of "rt".

        enable_shadow (bool): whether to enable shadow for lights. Defaults to False.

        sensor_configs (dict): configurations of sensors to override any environment defaults.
            If the key is one of sensor names (e.g. a camera), the config value will be applied to the corresponding sensor.
            Otherwise, the value will be applied to all sensors (but overridden by sensor-specific values). For possible configurations
            see the documentation see :doc:`the sensors documentation </user_guide/tutorials/sensors/index>`.

        human_render_camera_configs (dict): configurations of human rendering cameras to override any environment defaults. Similar usage as @sensor_configs.

        viewer_camera_configs (dict): configurations of the viewer camera in the GUI to override any environment defaults. Similar usage as @sensor_configs.

        robot_uids (Union[str, BaseAgent, List[Union[str, BaseAgent]]]): List of robots to instantiate and control in the environment.

        sim_config (Union[SimConfig, dict]): Configurations for simulation if used that override the environment defaults. If given
            a dictionary, it can just override specific attributes e.g. ``sim_config=dict(scene_config=dict(solver_iterations=25))``. If
            passing in a SimConfig object, while typed, will override every attribute including the task defaults. Some environments
            define their own recommended default sim configurations via the ``self._default_sim_config`` attribute that generally should not be
            completely overriden.

        reconfiguration_freq (int): How frequently to call reconfigure when environment is reset via `self.reset(...)`
            Generally for most users who are not building tasks this does not need to be changed. The default is 0, which means
            the environment reconfigures upon creation, and never again.

        sim_backend (str): By default this is "auto". If sim_backend is "auto", then if ``num_envs == 1``, we use the PhysX CPU sim backend, otherwise
            we use the PhysX GPU sim backend and automatically pick a GPU to use.
            Can also be "physx_cpu" or "physx_cuda" to force usage of a particular sim backend.
            To select a particular GPU to run the simulation on, you can pass "cuda:n" where n is the ID of the GPU,
            similar to the way PyTorch selects GPUs.
            Note that if this is "physx_cpu", num_envs can only be equal to 1.

        render_backend (str): By default this is "gpu". If render_backend is "gpu", then we auto select a GPU to render with.
            It can be "cuda:n" where n is the ID of the GPU to render with. If this is "cpu", then we render on the CPU.

        parallel_in_single_scene (bool): By default this is False. If True, rendered images and the GUI will show all objects in one view.
            This is only really useful for generating cool videos showing all environments at once but it is not recommended
            otherwise as it slows down simulation and rendering.

        enhanced_determinism (bool): By default this is False and env resets will reset the episode RNG only when a seed / seed list is given.
            If True, the environment will reset the episode RNG upon each reset regardless of whether a seed is provided.
            Generally enhanced_determinisim is not needed and users are recommended to pass seeds into the env reset function instead.
    """

    # fmt: off
    SUPPORTED_ROBOTS: List[Union[str, Tuple[str]]] = None
    """Override this to enforce which robots or tuples of robots together are supported in the task. During env creation,
    setting robot_uids auto loads all desired robots into the scene, but not all tasks are designed to support some robot setups"""
    SUPPORTED_OBS_MODES = ("state", "state_dict", "none", "sensor_data", "any_textures", "pointcloud")
    """The string observation modes the environment supports. Note that "none" and "any_texture" are special keys. none indicates no observation data is generated.
    "any_texture" indicates that any combination of image textures generated by cameras are supported e.g. rgb+depth, normal+segmentation, albedo+rgb+depth etc.
    For a full list of supported textures see """
    SUPPORTED_REWARD_MODES = ("normalized_dense", "dense", "sparse", "none")
    SUPPORTED_RENDER_MODES = ("human", "rgb_array", "sensors", "all")
    """The supported render modes. Human opens up a GUI viewer. rgb_array returns an rgb array showing the current environment state.
    sensors returns an rgb array but only showing all data collected by sensors as images put together"""

    metadata = {"render_modes": SUPPORTED_RENDER_MODES}

    scene: ManiSkillScene = None
    """the main scene, which manages all sub scenes. In CPU simulation there is only one sub-scene"""

    agent: BaseAgent

    action_space: gym.Space
    """the batched action space of the environment, which is also the action space of the agent"""
    single_action_space: gym.Space
    """the unbatched action space of the environment"""

    _sensors: Dict[str, BaseSensor]
    """all sensors configured in this environment"""
    _sensor_configs: Dict[str, BaseSensorConfig]
    """all sensor configurations parsed from self._sensor_configs and agent._sensor_configs"""
    _agent_sensor_configs: Dict[str, BaseSensorConfig]
    """all agent sensor configs parsed from agent._sensor_configs"""
    _human_render_cameras: Dict[str, Camera]
    """cameras used for rendering the current environment retrievable via `env.render_rgb_array()`. These are not used to generate observations"""
    _default_human_render_camera_configs: Dict[str, CameraConfig]
    """all camera configurations for cameras used for human render"""
    _human_render_camera_configs: Dict[str, CameraConfig]
    """all camera configurations parsed from self._human_render_camera_configs"""

    _hidden_objects: List[Union[Actor, Articulation]] = []
    """list of objects that are hidden during rendering when generating visual observations / running render_cameras()"""

    _main_rng: np.random.RandomState = None
    """main rng generator that generates episode seed sequences. For internal use only"""
    _batched_main_rng: BatchedRNG = None
    """the batched main RNG that generates episode seed sequences. For internal use only"""
    _main_seed: List[int] = None
    """main seed list for _main_rng and _batched_main_rng. _main_rng uses _main_seed[0]. For internal use only"""
    _episode_rng: np.random.RandomState = None
    """the numpy RNG that you can use to generate random numpy data. It is not recommended to use this. Instead use the _batched_episode_rng which helps ensure GPU and CPU simulation generate the same data with the same seeds."""
    _batched_episode_rng: BatchedRNG = None
    """the recommended batched episode RNG to generate random numpy data consistently between single and parallel environments"""
    _episode_seed: np.ndarray = None
    """episode seed list for _episode_rng and _batched_episode_rng. _episode_rng uses _episode_seed[0]."""
    _batched_rng_backend = "numpy:random_state"
    """the backend to use for the batched RNG"""
    _enhanced_determinism: bool = False
    """whether to reset the episode RNG upon each reset regardless of whether a seed is provided"""

    _parallel_in_single_scene: bool = False
    """whether all objects are placed in one scene for the purpose of rendering all objects together instead of in parallel"""

    _sim_device: sapien.Device = None
    """the sapien device object the simulation runs on"""

    _render_device: sapien.Device = None
    """the sapien device object the renderer runs on"""

    _viewer: Union[sapien.utils.Viewer, None] = None

    _sample_video_link: Optional[str] = None
    """a link to a sample video of the task. This is mostly used for automatic documentation generation"""

    def __init__(
        self,
        num_envs: int = 1,
        obs_mode: Optional[str] = None,
        reward_mode: Optional[str] = None,
        control_mode: Optional[str] = None,
        render_mode: Optional[str] = None,
        shader_dir: Optional[str] = None,
        enable_shadow: bool = False,
        sensor_configs: Optional[dict] = dict(),
        human_render_camera_configs: Optional[dict] = dict(),
        viewer_camera_configs: Optional[dict] = dict(),
        robot_uids: Union[str, BaseAgent, List[Union[str, BaseAgent]]] = None,
        sim_config: Union[SimConfig, dict] = dict(),
        reconfiguration_freq: Optional[int] = None,
        sim_backend: str = "auto",
        render_backend: str = "gpu",
        parallel_in_single_scene: bool = False,
        enhanced_determinism: bool = False,
    ):
        self._enhanced_determinism = enhanced_determinism

        self.num_envs = num_envs
        self.reconfiguration_freq = reconfiguration_freq if reconfiguration_freq is not None else 0
        self._reconfig_counter = 0
        if shader_dir is not None:
            logger.warn("shader_dir argument will be deprecated after ManiSkill v3.0.0 official release. Please use sensor_configs/human_render_camera_configs to set shaders.")
            sensor_configs |= dict(shader_pack=shader_dir)
            human_render_camera_configs |= dict(shader_pack=shader_dir)
            viewer_camera_configs |= dict(shader_pack=shader_dir)
        self._custom_sensor_configs = sensor_configs
        self._custom_human_render_camera_configs = human_render_camera_configs
        self._custom_viewer_camera_configs = viewer_camera_configs
        self._parallel_in_single_scene = parallel_in_single_scene
        self.robot_uids = robot_uids
        if isinstance(robot_uids, tuple) and len(robot_uids) == 1:
            self.robot_uids = robot_uids[0]
        if self.SUPPORTED_ROBOTS is not None:
            if self.robot_uids not in self.SUPPORTED_ROBOTS:
                logger.warn(f"{self.robot_uids} is not in the task's list of supported robots. Code may not run as intended")

        if sim_backend == "auto":
            if num_envs > 1:
                sim_backend = "physx_cuda"
            else:
                sim_backend = "physx_cpu"
        self.backend = parse_sim_and_render_backend(sim_backend, render_backend)
        # determine the sim and render devices
        self.device = self.backend.device
        self._sim_device = self.backend.sim_device
        self._render_device = self.backend.render_device
        if self.device.type == "cuda":
            if not physx.is_gpu_enabled():
                physx.enable_gpu()

        # raise a number of nicer errors
        if self.backend.sim_backend in CPU_SIM_BACKENDS and num_envs > 1:
            raise RuntimeError("""Cannot set the sim backend to 'cpu' and have multiple environments.
            If you want to do CPU sim backends and have environment vectorization you must use multi-processing across CPUs.
            This can be done via the gymnasium's AsyncVectorEnv API""")

        if shader_dir is not None:
            if "rt" == shader_dir[:2]:
                if num_envs > 1 and parallel_in_single_scene == False:
                    raise RuntimeError("""Currently you cannot run ray-tracing on more than one environment in a single process""")

        assert not parallel_in_single_scene or (obs_mode not in ["sensor_data", "pointcloud", "rgb", "depth", "rgbd"]), \
            "Parallel rendering from parallel cameras is only supported when the gui/viewer is not used. parallel_in_single_scene must be False if using parallel rendering. If True only state based observations are supported."

        if isinstance(sim_config, SimConfig):
            sim_config = sim_config.dict()
        merged_gpu_sim_config = self._default_sim_config.dict()
        common.dict_merge(merged_gpu_sim_config, sim_config)
        self.sim_config = dacite.from_dict(data_class=SimConfig, data=merged_gpu_sim_config, config=dacite.Config(strict=True))
        """the final sim config after merging user overrides with the environment default"""
        physx.set_gpu_memory_config(**self.sim_config.gpu_memory_config.dict())
        sapien.render.set_log_level(os.getenv("MS_RENDERER_LOG_LEVEL", "warn"))

        # Set simulation and control frequency
        self._sim_freq = self.sim_config.sim_freq
        self._control_freq = self.sim_config.control_freq
        assert self._sim_freq % self._control_freq == 0, f"sim_freq({self._sim_freq}) is not divisible by control_freq({self._control_freq})."
        self._sim_steps_per_control = self._sim_freq // self._control_freq

        # Observation mode
        if obs_mode is None:
            obs_mode = self.SUPPORTED_OBS_MODES[0]
        if obs_mode not in self.SUPPORTED_OBS_MODES:
            # we permit any combination of visual observation textures e.g. rgb+normal, depth+segmentation, etc.
            if "any_textures" in self.SUPPORTED_OBS_MODES:
                # the parse_visual_obs_mode_to_struct will check if the textures requested are valid
                pass
            else:
                raise NotImplementedError(f"Unsupported obs mode: {obs_mode}. Must be one of {self.SUPPORTED_OBS_MODES}")
        self._obs_mode = obs_mode
        self.obs_mode_struct = parse_obs_mode_to_struct(self._obs_mode)
        """dataclass describing what observation data is being requested by the user, detailing if state data is requested and what visual data is requested"""

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
        self._set_main_rng([2022 + i for i in range(self.num_envs)])
        self._elapsed_steps = (
            torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        )
        obs, _ = self.reset(seed=[2022 + i for i in range(self.num_envs)], options=dict(reconfigure=True))

        self._init_raw_obs = common.to_cpu_tensor(obs)
        """the raw observation returned by the env.reset (a cpu torch tensor/dict of tensors). Useful for future observation wrappers to use to auto generate observation spaces"""
        self._init_raw_state = common.to_cpu_tensor(self.get_state_dict())
        """the initial raw state returned by env.get_state. Useful for reconstructing state dictionaries from flattened state vectors"""

        if self.agent is not None:
            self.action_space = self.agent.action_space
            """the batched action space of the environment, which is also the action space of the agent"""
            self.single_action_space = self.agent.single_action_space
            """the unbatched action space of the environment"""
            self._orig_single_action_space = copy.deepcopy(self.single_action_space)
            """the original unbatched action space of the environment"""
        else:
            self.action_space = None
        # initialize the cached properties
        self.single_observation_space
        self.observation_space

    def update_obs_space(self, obs: torch.Tensor):
        """A convenient function to auto generate observation spaces if you modify them.
        Call this function if you modify the observations returned by env.step and env.reset via an observation wrapper.

        The recommended way to use this is in a observation wrapper is as so

        .. code-block:: python

            import gymnasium as gym
            from mani_skill.envs.sapien_env import BaseEnv
            class YourObservationWrapper(gym.ObservationWrapper):
                def __init__(self, env):
                    super().__init__(env)
                    self.base_env.update_obs_space(self.observation(self.base_env._init_raw_obs))
                @property
                def base_env(self) -> BaseEnv:
                    return self.env.unwrapped
                def observation(self, obs):
                    # your code for transforming the observation
        """
        self._init_raw_obs = obs
        del self.single_observation_space
        del self.observation_space
        self.single_observation_space
        self.observation_space

    @cached_property
    def single_observation_space(self) -> gym.Space:
        """the unbatched observation space of the environment"""
        return gym_utils.convert_observation_to_space(common.to_numpy(self._init_raw_obs), unbatched=True)

    @cached_property
    def observation_space(self) -> gym.Space:
        """the batched observation space of the environment"""
        return batch_space(self.single_observation_space, n=self.num_envs)

    @property
    def gpu_sim_enabled(self):
        """Whether the gpu simulation is enabled."""
        return self.scene.gpu_sim_enabled

    @property
    def _default_sim_config(self):
        return SimConfig()
    def _load_agent(self, options: dict, initial_agent_poses: Optional[Union[sapien.Pose, Pose]] = None, build_separate: bool = False):
        """
        loads the agent/controllable articulations into the environment. The default function provides a convenient way to setup the agent/robot by a robot_uid
        (stored in self.robot_uids) without requiring the user to have to write the robot building and controller code themselves. For more
        advanced use-cases you can override this function to have more control over the agent/robot building process.

        Args:
            options (dict): The options for the environment.
            initial_agent_poses (Optional[Union[sapien.Pose, Pose]]): The initial poses of the agent/robot. Providing these poses and ensuring they are picked such that
                they do not collide with objects if spawned there is highly recommended to ensure more stable simulation (the agent pose can be changed later during episode initialization).
            build_separate (bool): Whether to build the agent/robot separately. If True, the agent/robot will be built separately for each parallel environment and then merged
                together to be accessible under one view/object. This is useful for randomizing physical and visual properties of the agent/robot which is only permitted for
                articulations built separately in each environment.
        """
        agents = []
        robot_uids = self.robot_uids
        if not isinstance(initial_agent_poses, list):
            initial_agent_poses = [initial_agent_poses]
        if robot_uids == "none" or robot_uids == ("none", ):
            self.agent = None
            return
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
                    self.scene,
                    self._control_freq,
                    self._control_mode,
                    agent_idx=i if len(robot_uids) > 1 else None,
                    initial_pose=initial_agent_poses[i] if initial_agent_poses is not None else None,
                    build_separate=build_separate,
                )
                agents.append(agent)
        if len(agents) == 1:
            self.agent = agents[0]
        else:
            self.agent = MultiAgent(agents)

    @property
    def _default_sensor_configs(
        self,
    ) -> Union[
        BaseSensorConfig, Sequence[BaseSensorConfig], Dict[str, BaseSensorConfig]
    ]:
        """Add default (non-agent) sensors to the environment by returning sensor configurations. These can be overriden by the user at
        env creation time"""
        return []
    @property
    def _default_human_render_camera_configs(
        self,
    ) -> Union[
        CameraConfig, Sequence[CameraConfig], Dict[str, CameraConfig]
    ]:
        """Add default cameras for rendering when using render_mode='rgb_array'. These can be overriden by the user at env creation time """
        return []

    @property
    def _default_viewer_camera_configs(
        self,
    ) -> CameraConfig:
        """Default configuration for the viewer camera, controlling shader, fov, etc. By default if there is a human render camera called "render_camera" then the viewer will use that camera's pose."""
        return CameraConfig(uid="viewer", pose=sapien.Pose([0, 0, 1]), width=1920, height=1080, shader_pack="default", near=0.0, far=1000, fov=np.pi / 2)

    @property
    def sim_freq(self) -> int:
        """The frequency (Hz) of the simulation loop"""
        return self._sim_freq

    @property
    def control_freq(self):
        """The frequency (Hz) of the control loop"""
        return self._control_freq

    @property
    def sim_timestep(self):
        """The timestep (dt) of the simulation loop"""
        return 1.0 / self._sim_freq

    @property
    def control_timestep(self):
        """The timestep (dt) of the control loop"""
        return 1.0 / self._control_freq

    @property
    def control_mode(self) -> str:
        """The control mode of the agent"""
        return self.agent.control_mode

    @property
    def elapsed_steps(self) -> torch.Tensor:
        """The number of steps that have elapsed in the environment"""
        return self._elapsed_steps

    # ---------------------------------------------------------------------------- #
    # Observation
    # ---------------------------------------------------------------------------- #
    @property
    def obs_mode(self) -> str:
        """The current observation mode. This affects the observation returned by env.get_obs()"""
        return self._obs_mode

    def get_obs(self, info: Optional[Dict] = None):
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
            return dict()
        elif self._obs_mode == "state":
            state_dict = self._get_obs_state_dict(info)
            obs = common.flatten_state_dict(state_dict, use_torch=True, device=self.device)
        elif self._obs_mode == "state_dict":
            obs = self._get_obs_state_dict(info)
        elif self._obs_mode == "pointcloud":
            obs = self._get_obs_with_sensor_data(info)
            obs = sensor_data_to_pointcloud(obs, self._sensors)
        elif self._obs_mode == "sensor_data":
            # return raw texture data dependent on choice of shader
            obs = self._get_obs_with_sensor_data(info, apply_texture_transforms=False)
        else:
            obs = self._get_obs_with_sensor_data(info)

        # flatten parts of the state observation if requested
        if self.obs_mode_struct.state:
            if isinstance(obs, dict):
                data = dict(agent=obs.pop("agent"), extra=obs.pop("extra"))
                obs["state"] = common.flatten_state_dict(data, use_torch=True, device=self.device)
        return obs

    def _get_obs_state_dict(self, info: Dict):
        """Get (ground-truth) state-based observations."""
        return dict(
            agent=self._get_obs_agent(),
            extra=self._get_obs_extra(info),
        )

    def _get_obs_agent(self):
        """Get observations about the agent's state. By default it is proprioceptive observations which include qpos and qvel.
        Controller state is also included although most default controllers do not have any state."""
        return self.agent.get_proprioception()

    def _get_obs_extra(self, info: Dict):
        """Get task-relevant extra observations. Usually defined on a task by task basis"""
        return dict()

    def capture_sensor_data(self):
        """Capture data from all sensors (non-blocking)"""
        for sensor in self._sensors.values():
            sensor.capture()

    def get_sensor_images(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get image (RGB) visualizations of what sensors currently sense. This function calls self._get_obs_sensor_data() internally which automatically hides objects and updates the render"""
        return self.scene.get_sensor_images(self._get_obs_sensor_data())

    def get_sensor_params(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get all sensor parameters."""
        params = dict()
        for name, sensor in self._sensors.items():
            params[name] = sensor.get_params()
        return params

    def _get_obs_sensor_data(self, apply_texture_transforms: bool = True) -> dict:
        """get only data from sensors. Auto hides any objects that are designated to be hidden"""
        for obj in self._hidden_objects:
            obj.hide_visual()
        self.scene.update_render(update_sensors=True, update_human_render_cameras=False)
        self.capture_sensor_data()
        sensor_obs = dict()
        for name, sensor in self.scene.sensors.items():
            if isinstance(sensor, Camera):
                if self.obs_mode in ["state", "state_dict"]:
                    # normally in non visual observation modes we do not render sensor observations. But some users may want to render sensor data for debugging or various algorithms
                    sensor_obs[name] = sensor.get_obs(position=False, segmentation=False, apply_texture_transforms=apply_texture_transforms)
                else:
                    sensor_obs[name] = sensor.get_obs(
                        rgb=self.obs_mode_struct.visual.rgb,
                        depth=self.obs_mode_struct.visual.depth,
                        position=self.obs_mode_struct.visual.position,
                        segmentation=self.obs_mode_struct.visual.segmentation,
                        normal=self.obs_mode_struct.visual.normal,
                        albedo=self.obs_mode_struct.visual.albedo,
                        apply_texture_transforms=apply_texture_transforms
                    )
        # explicitly synchronize and wait for cuda kernels to finish
        # this prevents the GPU from making poor scheduling decisions when other physx code begins to run
        if self.backend.render_device.is_cuda():
            torch.cuda.synchronize()
        return sensor_obs
    def _get_obs_with_sensor_data(self, info: Dict, apply_texture_transforms: bool = True) -> dict:
        """Get the observation with sensor data"""
        return dict(
            agent=self._get_obs_agent(),
            extra=self._get_obs_extra(info),
            sensor_param=self.get_sensor_params(),
            sensor_data=self._get_obs_sensor_data(apply_texture_transforms),
        )

    @property
    def robot_link_names(self):
        """Get link ids for the robot. This is used for segmentation observations."""
        return self.agent.robot_link_names

    # -------------------------------------------------------------------------- #
    # Reward mode
    # -------------------------------------------------------------------------- #
    @property
    def reward_mode(self):
        return self._reward_mode

    def get_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        if self._reward_mode == "sparse":
            reward = self.compute_sparse_reward(obs=obs, action=action, info=info)
        elif self._reward_mode == "dense":
            reward = self.compute_dense_reward(obs=obs, action=action, info=info)
        elif self._reward_mode == "normalized_dense":
            reward = self.compute_normalized_dense_reward(
                obs=obs, action=action, info=info
            )
        elif self._reward_mode == "none":
            reward = torch.zeros((self.num_envs, ), dtype=torch.float, device=self.device)
        else:
            raise NotImplementedError(self._reward_mode)
        return reward

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """
        Computes the sparse reward. By default this function tries to use the success/fail information in
        returned by the evaluate function and gives +1 if success, -1 if fail, 0 otherwise"""
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
    def _reconfigure(self, options = dict()):
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

        self._load_agent(options)

        self._load_scene(options)
        self._load_lighting(options)

        self.scene._setup(enable_gpu=self.gpu_sim_enabled)
        # for GPU sim, we have to setup sensors after we call setup gpu in order to enable loading mounted sensors as they depend on GPU buffer data
        self._setup_sensors(options)
        if self.render_mode == "human" and self._viewer is None:
            self._viewer = create_viewer(self._viewer_camera_config)
        if self._viewer is not None:
            self._setup_viewer()
        self._reconfig_counter = self.reconfiguration_freq

        # delete various cached properties and reinitialize
        # TODO (stao): The code is 3 lines because you have to initialize it once somewhere...
        self.segmentation_id_map
        del self.segmentation_id_map
        self.segmentation_id_map

    def _after_reconfigure(self, options):
        """Add code here that should run immediately after self._reconfigure is called. The torch RNG context is still active so RNG is still
        seeded here by self._episode_seed. This is useful if you need to run something that only happens after reconfiguration but need the
        GPU initialized so that you can check e.g. collisons, poses etc."""

    def _load_scene(self, options: dict):
        """Loads all objects like actors and articulations into the scene. Called by `self._reconfigure`. Given options argument
        is the same options dictionary passed to the self.reset function"""

    # TODO (stao): refactor this into sensor API
    def _setup_sensors(self, options: dict):
        """Setup sensor configurations and the sensor objects in the scene. Called by `self._reconfigure`"""

        # First create all the configurations
        self._sensor_configs = dict()

        # Add task/external sensors
        self._sensor_configs.update(parse_camera_configs(self._default_sensor_configs))

        # Add agent sensors
        self._agent_sensor_configs = dict()
        if self.agent is not None:
            self._agent_sensor_configs = parse_camera_configs(self.agent._sensor_configs)
            self._sensor_configs.update(self._agent_sensor_configs)

        # Add human render camera configs
        self._human_render_camera_configs = parse_camera_configs(
            self._default_human_render_camera_configs
        )

        self._viewer_camera_config = parse_camera_configs(
            self._default_viewer_camera_configs
        )

        # Override camera configurations with user supplied configurations
        if self._custom_sensor_configs is not None:
            update_camera_configs_from_dict(
                self._sensor_configs, self._custom_sensor_configs
            )
        if self._custom_human_render_camera_configs is not None:
            update_camera_configs_from_dict(
                self._human_render_camera_configs,
                self._custom_human_render_camera_configs,
            )
        if self._custom_viewer_camera_configs is not None:
            update_camera_configs_from_dict(
                self._viewer_camera_config,
                self._custom_viewer_camera_configs,
            )
        self._viewer_camera_config = self._viewer_camera_config["viewer"]

        # Now we instantiate the actual sensor objects
        self._sensors = dict()

        for uid, sensor_config in self._sensor_configs.items():
            if uid in self._agent_sensor_configs:
                articulation = self.agent.robot
            else:
                articulation = None
            if isinstance(sensor_config, StereoDepthCameraConfig):
                sensor_cls = StereoDepthCamera
            elif isinstance(sensor_config, CameraConfig):
                sensor_cls = Camera
            self._sensors[uid] = sensor_cls(
                sensor_config,
                self.scene,
                articulation=articulation,
            )

        # Cameras for rendering only
        self._human_render_cameras = dict()
        for uid, camera_config in self._human_render_camera_configs.items():
            self._human_render_cameras[uid] = Camera(
                camera_config,
                self.scene,
            )

        self.scene.sensors = self._sensors
        self.scene.human_render_cameras = self._human_render_cameras

    def _load_lighting(self, options: dict):
        """Loads lighting into the scene. Called by `self._reconfigure`. If not overriden will set some simple default lighting"""

        shadow = self.enable_shadow
        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light(
            [1, 1, -1], [1, 1, 1], shadow=shadow, shadow_scale=5, shadow_map_size=2048
        )
        self.scene.add_directional_light([0, 0, -1], [1, 1, 1])
    # -------------------------------------------------------------------------- #
    # Reset
    # -------------------------------------------------------------------------- #
    def reset(self, seed: Union[None, int, list[int]] = None, options: Union[None, dict] = None):
        """Reset the ManiSkill environment with given seed(s) and options. Typically seed is either None (for unseeded reset) or an int (seeded reset).
        For GPU parallelized environments you can also pass a list of seeds for each parallel environment to seed each one separately.

        If options["env_idx"] is given, will only reset the selected parallel environments. If
        options["reconfigure"] is True, will call self._reconfigure() which deletes the entire physx scene and reconstructs everything.
        Users building custom tasks generally do not need to override this function.

        Returns the first observation and a info dictionary. The info dictionary is of type


        .. highlight:: python
        .. code-block:: python

            {
                "reconfigure": bool # (True if the env reconfigured. False otherwise)
            }



        Note that ManiSkill always holds two RNG states, a main RNG, and an episode RNG. The main RNG is used purely to sample an episode seed which
        helps with reproducibility of episodes and is for internal use only. The episode RNG is used by the environment/task itself to
        e.g. randomize object positions, randomize assets etc. Episode RNG is accessible by using `self._batched_episode_rng` which is numpy based and `torch.rand`
        which can be used to generate random data on the GPU directly and is seeded. Note that it is recommended to use `self._batched_episode_rng`
        if you need to ensure during reconfiguration the same objects are loaded. Reproducibility and seeding when there is GPU and CPU simulation can be tricky and we recommend reading
        the documentation for more recommendations and details on RNG https://maniskill.readthedocs.io/en/latest/user_guide/concepts/rng.html

        Upon environment creation via gym.make, the main RNG is set with fixed seeds of 2022 to 2022 + num_envs - 1 (seed is just 2022 if there is only one environment)
        During each reset call, if seed is None, main RNG is unchanged and an episode seed is sampled from the main RNG to create the episode RNG.
        If seed is not None, main RNG is set to that seed and the episode seed is also set to that seed. This design means the main RNG determines
        the episode RNG deterministically.

        """
        if options is None:
            options = dict()
        reconfigure = options.get("reconfigure", False)
        reconfigure = reconfigure or (
            self._reconfig_counter == 0 and self.reconfiguration_freq != 0
        )
        if "env_idx" in options:
            env_idx = options["env_idx"]
            if len(env_idx) != self.num_envs and reconfigure:
                raise RuntimeError("Cannot do a partial reset and reconfigure the environment. You must do one or the other.")
        else:
            env_idx = torch.arange(0, self.num_envs, device=self.device)

        self._set_main_rng(seed)

        if reconfigure:
            self._set_episode_rng(seed if seed is not None else self._batched_main_rng.randint(2**31), env_idx)
            with torch.random.fork_rng():
                torch.manual_seed(seed=self._episode_seed[0])
                self._reconfigure(options)
                self._after_reconfigure(options)
            # Set the episode rng again after reconfiguration to guarantee seed reproducibility
            self._set_episode_rng(self._episode_seed, env_idx)
        else:
            self._set_episode_rng(seed, env_idx)

        # TODO (stao): Reconfiguration when there is partial reset might not make sense and certainly broken here now.
        # Solution to resolve that would be to ensure tasks that do reconfigure more than once are single-env only / cpu sim only
        # or disable partial reset features explicitly for tasks that have a reconfiguration frequency
        self.scene._reset_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.scene._reset_mask[env_idx] = True
        self._elapsed_steps[env_idx] = 0

        self._clear_sim_state()
        if self.reconfiguration_freq != 0:
            self._reconfig_counter -= 1

        if self.agent is not None:
            self.agent.reset()

        if seed is not None or self._enhanced_determinism:
            with torch.random.fork_rng():
                torch.manual_seed(self._episode_seed[0])
                self._initialize_episode(env_idx, options)
        else:
            self._initialize_episode(env_idx, options)
        # reset the reset mask back to all ones so any internal code in maniskill can continue to manipulate all scenes at once as usual
        self.scene._reset_mask = torch.ones(
            self.num_envs, dtype=bool, device=self.device
        )
        if self.gpu_sim_enabled:
            # ensure all updates to object poses and configurations are applied on GPU after task initialization
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()

        # we reset controllers here because some controllers depend on the agent/articulation qpos/poses
        if self.agent is not None:
            if isinstance(self.agent.controller, dict):
                for controller in self.agent.controller.values():
                    controller.reset()
            else:
                self.agent.controller.reset()

        info = self.get_info()
        obs = self.get_obs(info)

        info["reconfigure"] = reconfigure
        return obs, info

    def _set_main_rng(self, seed):
        """Set the main random generator which is only used to set the seed of the episode RNG to improve reproducibility.

        Note that while _set_main_rng and _set_episode_rng are setting a seed and numpy random state, when using GPU sim
        parallelization it is highly recommended to use torch random functions as they will make things run faster. The use
        of torch random functions when building tasks in ManiSkill are automatically seeded via `torch.random.fork`
        """
        if seed is None:
            if self._main_seed is not None:
                return
            seed = np.random.RandomState().randint(2**31, size=(self.num_envs,))
        if not np.iterable(seed):
            seed = [seed]
        self._main_seed = seed
        self._main_rng = np.random.RandomState(self._main_seed[0])
        if len(self._main_seed) == 1 and self.num_envs > 1:
            self._main_seed = self._main_seed + np.random.RandomState(self._main_seed[0]).randint(2**31, size=(self.num_envs - 1,)).tolist()
        self._batched_main_rng = BatchedRNG.from_seeds(self._main_seed, backend=self._batched_rng_backend)

    def _set_episode_rng(self, seed: Union[None, list[int]], env_idx: torch.Tensor):
        """Set the random generator for current episode."""
        if seed is not None or self._enhanced_determinism:
            env_idx = common.to_numpy(env_idx)
            if seed is None:
                self._episode_seed[env_idx] = self._batched_main_rng[env_idx].randint(2**31)
            else:
                if not np.iterable(seed):
                    seed = [seed]
                self._episode_seed = common.to_numpy(seed, dtype=np.int64)
                if len(self._episode_seed) == 1 and self.num_envs > 1:
                    self._episode_seed = np.concatenate((self._episode_seed, np.random.RandomState(self._episode_seed[0]).randint(2**31, size=(self.num_envs - 1,))))
            # we keep _episode_rng for backwards compatibility but recommend using _batched_episode_rng for randomization
            if seed is not None or self._batched_episode_rng is None:
                self._batched_episode_rng = BatchedRNG.from_seeds(self._episode_seed, backend=self._batched_rng_backend)
            else:
                self._batched_episode_rng[env_idx] = BatchedRNG.from_seeds(self._episode_seed[env_idx], backend=self._batched_rng_backend)
            self._episode_rng = self._batched_episode_rng[0]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """Initialize the episode, e.g., poses of actors and articulations, as well as task relevant data like randomizing
        goal positions
        """

    def _clear_sim_state(self):
        """Clear simulation state (velocities)"""
        for actor in self.scene.actors.values():
            if actor.px_body_type == "dynamic":
                actor.set_linear_velocity(torch.zeros(3, device=self.device))
                actor.set_angular_velocity(torch.zeros(3, device=self.device))
        for articulation in self.scene.articulations.values():
            articulation.set_qvel(torch.zeros(articulation.max_dof, device=self.device))
            articulation.set_root_linear_velocity(torch.zeros(3, device=self.device))
            articulation.set_root_angular_velocity(torch.zeros(3, device=self.device))
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene._gpu_fetch_all()
            # TODO (stao): This may be an unnecessary fetch and apply.

    # -------------------------------------------------------------------------- #
    # Step
    # -------------------------------------------------------------------------- #

    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):
        """
        Take a step through the environment with an action. Actions are automatically clipped to the action space.

        If ``action`` is None, the environment will proceed forward in time without sending any actions/control signals to the agent
        """
        action = self._step_action(action)
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

        return (
            obs,
            reward,
            terminated,
            torch.zeros(self.num_envs, dtype=bool, device=self.device),
            info,
        )

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
                if action["control_mode"] != self.agent.control_mode:
                    self.agent.set_control_mode(action["control_mode"])
                    self.agent.controller.reset()
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

        if set_action:
            if self.num_envs == 1 and action_is_unbatched:
                action = common.batch(action)
            self.agent.set_action(action)
            if self._sim_device.is_cuda():
                self.scene.px.gpu_apply_articulation_target_position()
                self.scene.px.gpu_apply_articulation_target_velocity()
        self._before_control_step()
        for _ in range(self._sim_steps_per_control):
            if self.agent is not None:
                self.agent.before_simulation_step()
            self._before_simulation_step()
            self.scene.step()
            self._after_simulation_step()
        self._after_control_step()
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()
        return action

    def evaluate(self) -> dict:
        """
        Evaluate whether the environment is currently in a success state by returning a dictionary with a "success" key or
        a failure state via a "fail" key

        This function may also return additional data that has been computed (e.g. is the robot grasping some object) that may be
        reused when generating observations and rewards.

        By default if not overriden this function returns an empty dictionary
        """
        return dict()

    def get_info(self) -> dict:
        """
        Get info about the current environment state, include elapsed steps and evaluation information
        """
        info = dict(
            elapsed_steps=self.elapsed_steps
            if not self.gpu_sim_enabled
            else self._elapsed_steps.clone()
        )
        info.update(self.evaluate())
        return info

    def _before_control_step(self):
        """Code that runs before each action has been taken via env.step(action).
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
        physx.set_shape_config(contact_offset=self.sim_config.scene_config.contact_offset, rest_offset=self.sim_config.scene_config.rest_offset)
        physx.set_body_config(solver_position_iterations=self.sim_config.scene_config.solver_position_iterations, solver_velocity_iterations=self.sim_config.scene_config.solver_velocity_iterations, sleep_threshold=self.sim_config.scene_config.sleep_threshold)
        physx.set_scene_config(gravity=self.sim_config.scene_config.gravity, bounce_threshold=self.sim_config.scene_config.bounce_threshold, enable_pcm=self.sim_config.scene_config.enable_pcm, enable_tgs=self.sim_config.scene_config.enable_tgs, enable_ccd=self.sim_config.scene_config.enable_ccd, enable_enhanced_determinism=self.sim_config.scene_config.enable_enhanced_determinism, enable_friction_every_iteration=self.sim_config.scene_config.enable_friction_every_iteration, cpu_workers=self.sim_config.scene_config.cpu_workers )
        physx.set_default_material(**self.sim_config.default_materials_config.dict())

    def _setup_scene(self):
        """Setup the simulation scene instance.
        The function should be called in reset(). Called by `self._reconfigure`"""
        self._set_scene_config()
        if self._sim_device.is_cuda():
            physx_system = physx.PhysxGpuSystem(device=self._sim_device)
            # Create the scenes in a square grid
            sub_scenes = []
            scene_grid_length = int(np.ceil(np.sqrt(self.num_envs)))
            for scene_idx in range(self.num_envs):
                scene_x, scene_y = (
                    scene_idx % scene_grid_length - scene_grid_length // 2,
                    scene_idx // scene_grid_length - scene_grid_length // 2,
                )
                scene = sapien.Scene(
                    systems=[physx_system, sapien.render.RenderSystem(self._render_device)]
                )
                physx_system.set_scene_offset(
                    scene,
                    [
                        scene_x * self.sim_config.spacing,
                        scene_y * self.sim_config.spacing,
                        0,
                    ],
                )
                sub_scenes.append(scene)
        else:
            physx_system = physx.PhysxCpuSystem()
            sub_scenes = [
                sapien.Scene([physx_system, sapien.render.RenderSystem(self._render_device)])
            ]
        # create a "global" scene object that users can work with that is linked with all other scenes created
        self.scene = ManiSkillScene(
            sub_scenes,
            sim_config=self.sim_config,
            device=self.device,
            parallel_in_single_scene=self._parallel_in_single_scene,
            backend=self.backend
        )
        self.scene.px.timestep = 1.0 / self._sim_freq

    def _clear(self):
        """Clear the simulation scene instance and other buffers.
        The function can be called in reset() before a new scene is created.
        Called by `self._reconfigure` and when the environment is closed/deleted
        """
        self._close_viewer()
        self.agent = None
        self._sensors = dict()
        self._human_render_cameras = dict()
        self.scene = None
        self._hidden_objects = []
        gc.collect() # force gc to collect which releases most GPU memory

    def close(self):
        self._clear()

    def _close_viewer(self):
        if self._viewer is None:
            return
        self._viewer.close()
        self._viewer = None

    @cached_property
    def segmentation_id_map(self):
        """
        Returns a dictionary mapping every ID to the appropriate Actor or Link object
        """
        res = dict()
        for actor in self.scene.actors.values():
            res[actor._objs[0].per_scene_id] = actor
        for art in self.scene.articulations.values():
            for link in art.links:
                res[link._objs[0].entity.per_scene_id] = link
        return res

    def add_to_state_dict_registry(self, object: Union[Actor, Articulation]):
        self.scene.add_to_state_dict_registry(object)
    def remove_from_state_dict_registry(self, object: Union[Actor, Articulation]):
        self.scene.remove_from_state_dict_registry(object)

    def get_state_dict(self):
        """
        Get environment state dictionary. Override to include task information (e.g., goal)
        """
        return self.scene.get_sim_state()

    def get_state(self):
        """
        Get environment state as a flat vector, which is just a ordered flattened version of the state_dict.

        Users should not override this function
        """
        return common.flatten_state_dict(self.get_state_dict(), use_torch=True)

    def set_state_dict(self, state: Dict, env_idx: torch.Tensor = None):
        """
        Set environment state with a state dictionary. Override to include task information (e.g., goal)

        Note that it is recommended to keep around state dictionaries as opposed to state vectors. With state vectors we assume
        the order of data in the vector is the same exact order that would be returned by flattening the state dictionary you get from
        `env.get_state_dict()` or the result of `env.get_state()`
        """
        self.scene.set_sim_state(state, env_idx)
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()

    def set_state(self, state: Array, env_idx: torch.Tensor = None):
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
        self.set_state_dict(state_dict, env_idx)

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
        self._viewer.set_scene(self.scene.sub_scenes[0])
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
        """render the environment by opening a GUI viewer. This also returns the viewer object. Any objects registered in the _hidden_objects list will be shown"""
        for obj in self._hidden_objects:
            obj.show_visual()
        if self._viewer is None:
            self._viewer = create_viewer(self._viewer_camera_config)
            self._setup_viewer()
        if self.gpu_sim_enabled and self.scene._gpu_sim_initialized:
            self.scene.px.sync_poses_gpu_to_cpu()
        self._viewer.render()
        for obj in self._hidden_objects:
            obj.hide_visual()
        return self._viewer

    def render_rgb_array(self, camera_name: str = None):
        """Returns an RGB array / image of size (num_envs, H, W, 3) of the current state of the environment.
        This is captured by any of the registered human render cameras. If a camera_name is given, only data from that camera is returned.
        Otherwise all camera data is captured and returned as a single batched image. Any objects registered in the _hidden_objects list will be shown"""
        for obj in self._hidden_objects:
            obj.show_visual()
        self.scene.update_render(update_sensors=False, update_human_render_cameras=True)
        images = []
        render_images = self.scene.get_human_render_camera_images(camera_name)
        for image in render_images.values():
            images.append(image)
        if len(images) == 0:
            return None
        if len(images) == 1:
            return images[0]
        for obj in self._hidden_objects:
            obj.hide_visual()
        return tile_images(images)

    def render_sensors(self):
        """
        Renders all sensors that the agent can use and see and displays them in a human readable image format.
        Any objects registered in the _hidden_objects list will not be shown
        """
        images = []
        sensor_images = self.get_sensor_images()
        for image in sensor_images.values():
            for img in image.values():
                images.append(img)
        return tile_images(images)

    def render_all(self):
        """Renders all human render cameras and sensors together"""
        images = []
        for obj in self._hidden_objects:
            obj.show_visual()
        self.scene.update_render(update_sensors=True, update_human_render_cameras=True)
        render_images = self.scene.get_human_render_camera_images()
        # note that get_sensor_images function will update the render and hide objects itself
        sensor_images = self.get_sensor_images()
        for image in render_images.values():
            images.append(image)
        for image in sensor_images.values():
            for img in image.values():
                images.append(img)
        return tile_images(images)

    def render(self):
        """
        Either opens a viewer if ``self.render_mode`` is "human", or returns an array that you can use to save videos.

        If ``self.render_mode`` is "rgb_array", usually a higher quality image is rendered for the purpose of viewing only.

        If ``self.render_mode`` is "sensors", all visual observations the agent can see is provided

        If ``self.render_mode`` is "all", this is then a combination of "rgb_array" and "sensors"
        """
        if self.render_mode is None:
            raise RuntimeError("render_mode is not set.")
        if self.render_mode == "human":
            return self.render_human()
        elif self.render_mode == "rgb_array":
            res = self.render_rgb_array()
            return res
        elif self.render_mode == "sensors":
            res = self.render_sensors()
            return res
        elif self.render_mode == "all":
            return self.render_all()
        else:
            raise NotImplementedError(f"Unsupported render mode {self.render_mode}.")

    # TODO (stao): re implement later
    # ---------------------------------------------------------------------------- #
    # Advanced
    # ---------------------------------------------------------------------------- #

    # def gen_scene_pcd(self, num_points: int = int(1e5)) -> np.ndarray:
    #     """Generate scene point cloud for motion planning, excluding the robot"""
    #     meshes = []
    #     articulations = self.scene.get_all_articulations()
    #     if self.agent is not None:
    #         articulations.pop(articulations.index(self.agent.robot))
    #     for articulation in articulations:
    #         articulation_mesh = merge_meshes(get_articulation_meshes(articulation))
    #         if articulation_mesh:
    #             meshes.append(articulation_mesh)

    #     for actor in self.scene.get_all_actors():
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


    # Printing metrics/info
    def print_sim_details(self):
        """Debug tool to call to simply print a bunch of details about the running environment, including the task ID, number of environments, sim backend, etc."""
        sensor_settings_str = []
        for uid, cam in self._sensors.items():
            if isinstance(cam, Camera):
                config = cam.config
                sensor_settings_str.append(f"RGBD({config.width}x{config.height})")
        sensor_settings_str = ", ".join(sensor_settings_str)
        sim_backend = self.backend.sim_backend
        print(
        "# -------------------------------------------------------------------------- #"
        )
        print(
            f"Task ID: {self.spec.id}, {self.num_envs} parallel environments, sim_backend={sim_backend}"
        )
        print(
            f"obs_mode={self.obs_mode}, control_mode={self.control_mode}"
        )
        print(
            f"render_mode={self.render_mode}, sensor_details={sensor_settings_str}"
        )
        print(
            f"sim_freq={self.sim_freq}, control_freq={self.control_freq}"
        )
        print(f"observation space: {self.observation_space}")
        print(f"(single) action space: {self.single_action_space}")
        print(
            "# -------------------------------------------------------------------------- #"
        )
