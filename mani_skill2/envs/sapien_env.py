import copy
import os
from collections import OrderedDict
from typing import Any, Dict, List, Sequence, Type, Union

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

from mani_skill2 import logger
from mani_skill2.agents.base_agent import BaseAgent
from mani_skill2.agents.robots import ROBOTS
from mani_skill2.envs.scene import ManiSkillScene
from mani_skill2.sensors.base_sensor import BaseSensor, BaseSensorConfig
from mani_skill2.sensors.camera import (
    Camera,
    CameraConfig,
    parse_camera_cfgs,
    update_camera_cfgs_from_dict,
)
from mani_skill2.sensors.depth_camera import StereoDepthCamera, StereoDepthCameraConfig
from mani_skill2.utils.common import convert_observation_to_space, flatten_state_dict
from mani_skill2.utils.geometry.trimesh_utils import (
    get_articulation_meshes,
    get_component_meshes,
    merge_meshes,
)
from mani_skill2.utils.sapien_utils import (
    batch,
    get_obj_by_type,
    to_numpy,
    to_tensor,
    unbatch,
)
from mani_skill2.utils.structs.types import Array
from mani_skill2.utils.visualization.misc import observations_to_images, tile_images


class BaseEnv(gym.Env):
    """Superclass for ManiSkill environments.

    Args:
        num_envs: number of parallel environments to run. By default this is 1, which means a CPU simulation is used. If greater than 1,
            then we initialize the GPU simulation setup. Note that not all environments are faster when simulated on the GPU due to limitations of
            GPU simulations. For example, environments with many moving objects are better simulated by parallelizing across CPUs.
        gpu_sim_backend: The GPU simulation backend to use (only used if the given num_envs argument is > 1). This affects the type of tensor
            returned by the environment for e.g. observations and rewards. Can be "torch" or "jax". Default is "torch"
        obs_mode: observation mode registered in @SUPPORTED_OBS_MODES.
        reward_mode: reward mode registered in @SUPPORTED_REWARD_MODES.
        control_mode: control mode of the agent.
            "*" represents all registered controllers, and the action space will be a dict.
        render_mode: render mode registered in @SUPPORTED_RENDER_MODES.
        sim_freq (int): simulation frequency (Hz)
        control_freq (int): control frequency (Hz)
        renderer (str): type of renderer. "sapien" or "client".
        renderer_kwargs (dict): kwargs to initialize the renderer.
            Example kwargs for `SapienRenderer` (renderer_type=='sapien'):
            - offscreen_only: tell the renderer the user does not need to present onto a screen.
            - device (str): GPU device for renderer, e.g., 'cuda:x'.
        shader_dir (str): shader directory. Defaults to "default".
            "default" and "rt" are built-in options with SAPIEN. Other options are user-defined.
        render_config (dict): kwargs to configure the renderer. Only for `SapienRenderer`.
            See `sapien.RenderConfig` for more details.
        enable_shadow (bool): whether to enable shadow for lights. Defaults to False.
        sensor_cfgs (dict): configurations of sensors. See notes for more details.
        render_camera_cfgs (dict): configurations of rendering cameras. Similar usage as @camera_cfgs.
        gpu_sim_cfgs (dict): Configurations for GPU simulation if used.

    Note:
        `sensor_cfgs` is used to update environement-specific sensor configurations.
        If the key is one of sensor names (e.g. a camera), the value will be applied to the corresponding sensor.
        Otherwise, the value will be applied to all sensors (but overridden by sensor-specific values).
    """

    # fmt: off
    SUPPORTED_OBS_MODES = ("state", "state_dict", "none", "image")
    SUPPORTED_REWARD_MODES = ("normalized_dense", "dense", "sparse")
    SUPPORTED_RENDER_MODES = ("human", "rgb_array", "cameras")
    # fmt: on

    metadata = {"render_modes": SUPPORTED_RENDER_MODES}

    physx_system: Union[sapien.physx.PhysxCpuSystem, sapien.physx.PhysxGpuSystem] = None

    # the main scene, which manages all sub scenes. In CPU simulation there is only one sub-scene
    _scene: ManiSkillScene = None

    _agent_cls: Type[BaseAgent]
    agent: BaseAgent
    _sensors: Dict[str, BaseSensor]
    _sensor_cfgs: Dict[str, BaseSensorConfig]
    _agent_camera_cfgs: Dict[str, CameraConfig]

    # render cameras are sensors that are not part of any observations
    _render_cameras: Dict[str, Camera]
    _render_camera_cfgs: Dict[str, CameraConfig]

    def __init__(
        self,
        num_envs: int = 1,
        obs_mode: str = None,
        reward_mode: str = None,
        control_mode: str = None,
        render_mode: str = None,
        sim_freq: int = 500,
        control_freq: int = 20,
        renderer: str = "sapien",
        renderer_kwargs: dict = None,
        shader_dir: str = "default",
        render_config: dict = None,
        enable_shadow: bool = False,
        sensor_cfgs: dict = None,
        render_camera_cfgs: dict = None,
        robot_uid: Union[str, BaseAgent] = None,
        gpu_sim_cfgs: dict = dict(spacing=10),
    ):
        # Create SAPIEN engine

        self._scene: ManiSkillScene = None
        self.num_envs = num_envs
        if num_envs > 1:
            if not sapien.physx.is_gpu_enabled():
                sapien.physx.enable_gpu()
                sapien.set_cuda_tensor_backend("torch")
            self.gpu_sim_cfgs = gpu_sim_cfgs
            self.device = torch.device(
                "cuda"
            )  # TODO (stao): fix this for multi gpu support?
        else:
            self.device = torch.device("cpu")
        # TODO(jigu): Change to `warning` after lighting in VecEnv is fixed.
        # TODO Ms2 set log level. What to do now?
        # self._engine.set_log_level(os.getenv("MS2_SIM_LOG_LEVEL", "error"))

        # Create SAPIEN renderer
        self._renderer_type = renderer
        if renderer_kwargs is None:
            renderer_kwargs = {}
        if self._renderer_type == "sapien":
            # TODO (stao): we need to deprecate use to the self._renderer
            self._renderer = sapien.SapienRenderer(**renderer_kwargs)
            if shader_dir == "default":
                sapien.render.set_camera_shader_dir("minimal")
                sapien.render.set_picture_format("Color", "r8g8b8a8unorm")
                sapien.render.set_picture_format("ColorRaw", "r8g8b8a8unorm")
                sapien.render.set_picture_format(
                    "PositionSegmentation", "r16g16b16a16sint"
                )
            elif shader_dir == "rt":
                sapien.render.set_camera_shader_dir("rt")
                sapien.render.set_viewer_shader_dir("rt")
                sapien.render.set_ray_tracing_samples_per_pixel(32)
                sapien.render.set_ray_tracing_path_depth(16)
                sapien.render.set_ray_tracing_denoiser(
                    "optix"
                )  # TODO "optix or oidn?" previous value was just True
            elif shader_dir == "rt-fast":
                sapien.render.set_camera_shader_dir("rt")
                sapien.render.set_viewer_shader_dir("rt")
                sapien.render.set_ray_tracing_samples_per_pixel(2)
                sapien.render.set_ray_tracing_path_depth(1)
                sapien.render.set_ray_tracing_denoiser("optix")
            sapien.render.set_log_level(os.getenv("MS2_RENDERER_LOG_LEVEL", "warn"))

        # TODO (stao): what here?
        # elif self._renderer_type == "client":
        #     self._renderer = sapien.RenderClient(**renderer_kwargs)
        #     # TODO(jigu): add `set_log_level` for RenderClient?
        # else:
        #     raise NotImplementedError(self._renderer_type)

        # Set simulation and control frequency
        self._sim_freq = sim_freq
        self._control_freq = control_freq
        if sim_freq % control_freq != 0:
            logger.warning(
                f"sim_freq({sim_freq}) is not divisible by control_freq({control_freq}).",
            )
        self._sim_steps_per_control = sim_freq // control_freq

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

        if robot_uid is not None:
            if isinstance(robot_uid, type(BaseAgent)):
                self._agent_cls = robot_uid
                robot_uid = self._agent_cls.uid
            else:
                self._agent_cls = ROBOTS[robot_uid]
            self.robot_uid = robot_uid

        # NOTE(jigu): Agent and camera configurations should not change after initialization.
        self._configure_sensors()
        self._configure_render_cameras()
        # Override camera configurations
        if sensor_cfgs is not None:
            update_camera_cfgs_from_dict(self._sensor_cfgs, sensor_cfgs)
        if render_camera_cfgs is not None:
            update_camera_cfgs_from_dict(self._render_camera_cfgs, render_camera_cfgs)

        # Lighting
        self.enable_shadow = enable_shadow

        # Use a fixed (main) seed to enhance determinism
        self._main_seed = None
        self._set_main_rng(2022)
        obs, _ = self.reset(seed=2022, options=dict(reconfigure=True))
        if physx.is_gpu_enabled():
            obs = to_numpy(obs)
        # TODO handle constructing single obs space from a batched result.

        if num_envs > 1:
            self.single_observation_space = convert_observation_to_space(
                obs, unbatched=True
            )
            self.observation_space = batch_space(
                self.single_observation_space, n=self.num_envs
            )
        else:
            self.observation_space = convert_observation_to_space(obs)
        if self._obs_mode == "image":
            image_obs_space = self.observation_space.spaces["image"]
            for uid, camera in self._sensors.items():
                image_obs_space.spaces[uid] = camera.observation_space

        self.action_space = self.agent.action_space
        self.single_action_space = copy.deepcopy(self.action_space)
        if num_envs > 1:
            self.agent.controller._batch_action_space()
            self.action_space = self.agent.action_space

    def _load_agent(self):
        agent_cls: Type[BaseAgent] = self._agent_cls
        self.agent = agent_cls(self._scene, self._control_freq, self._control_mode)
        # set_articulation_render_material(self.agent.robot, specular=0.9, roughness=0.3)
        self.agent.set_control_mode(self.agent._default_control_mode)

    def _configure_sensors(self):
        self._sensor_cfgs = OrderedDict()

        # Add task/external sensors
        self._sensor_cfgs.update(parse_camera_cfgs(self._register_sensors()))

        # Add agent sensors
        self._agent_camera_cfgs = OrderedDict()
        self._agent_camera_cfgs = parse_camera_cfgs(self._agent_cls.sensor_configs)
        self._sensor_cfgs.update(self._agent_camera_cfgs)

    def _register_sensors(
        self,
    ) -> Union[
        BaseSensorConfig, Sequence[BaseSensorConfig], Dict[str, BaseSensorConfig]
    ]:
        """Register (non-agent) sensors for the environment."""
        return []

    def _configure_render_cameras(self):
        self._render_camera_cfgs = parse_camera_cfgs(self._register_render_cameras())

    def _register_render_cameras(
        self,
    ) -> Union[
        BaseSensorConfig, Sequence[BaseSensorConfig], Dict[str, BaseSensorConfig]
    ]:
        """Register cameras for rendering."""
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

    def get_obs(self):
        """
        Return the current observation of the environment
        """
        squeeze_dims = self.num_envs == 1
        if self._obs_mode == "none":
            # Some cases do not need observations, e.g., MPC
            return OrderedDict()
        elif self._obs_mode == "state":
            state_dict = self._get_obs_state_dict()
            obs = flatten_state_dict(state_dict, squeeze_dims=squeeze_dims)
        elif self._obs_mode == "state_dict":
            obs = self._get_obs_state_dict()
        elif self._obs_mode == "image":
            obs = self._get_obs_images()
        else:
            raise NotImplementedError(self._obs_mode)
        return obs

    def _get_obs_state_dict(self):
        """Get (ground-truth) state-based observations."""
        return OrderedDict(
            agent=self._get_obs_agent(),
            extra=self._get_obs_extra(),
        )

    def _get_obs_agent(self):
        """Get observations from the agent's sensors, e.g., proprioceptive sensors."""
        return self.agent.get_proprioception()

    def _get_obs_extra(self):
        """Get task-relevant extra observations."""
        return OrderedDict()

    def update_render(self):
        """Update renderer(s). This function should be called before any rendering,
        to sync simulator and renderer."""
        # TODO (stao): note that update_render has some overhead. Currently when using image observation mode + using render() for recording videos
        # this is called twice

        # TODO (stao): We might want to factor out some of this code below
        self._scene.update_render()

    def capture_sensor_data(self):
        """Take pictures from all cameras and sensors (non-blocking)"""
        for cam in self._sensors.values():
            if isinstance(cam, Camera):
                cam.take_picture()
            else:
                raise NotImplementedError(
                    "Other modalities of sensor data not implemented yet"
                )

    def get_images(self) -> Dict[str, Dict[str, np.ndarray]]:
        # TODO (stao): support general sensors later.
        """Get (raw) images from all cameras (blocking)."""
        images = OrderedDict()
        for name, cam in self._sensors.items():
            if isinstance(cam, Camera):
                images[name] = cam.get_images()
        return images

    def get_camera_params(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Get camera parameters from all cameras."""
        params = OrderedDict()
        for name, cam in self._sensors.items():
            params[name] = cam.get_params()
        return params

    def _get_obs_images(self) -> OrderedDict:
        self.update_render()
        self.capture_sensor_data()
        return OrderedDict(
            agent=self._get_obs_agent(),
            extra=self._get_obs_extra(),
            camera_param=self.get_camera_params(),
            image=self.get_images(),
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

    def get_reward(self, obs: Any, action: Array, info: Dict):
        if self._reward_mode == "sparse":
            reward = info["success"]
        elif self._reward_mode == "dense":
            reward = self.compute_dense_reward(obs=obs, action=action, info=info)
        elif self._reward_mode == "normalized_dense":
            reward = self.compute_normalized_dense_reward(
                obs=obs, action=action, info=info
            )
        else:
            raise NotImplementedError(self._reward_mode)
        return reward

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        raise NotImplementedError

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        raise NotImplementedError

    # -------------------------------------------------------------------------- #
    # Reconfigure
    # -------------------------------------------------------------------------- #
    def reconfigure(self):
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
        self._load_actors()
        self._load_articulations()

        self._setup_lighting()

        # Cache entites and articulations
        self._actors = self.get_actors()
        self._articulations = self.get_articulations()
        if sapien.physx.is_gpu_enabled():
            self._scene._setup_gpu()
            self.device = self.physx_system.cuda_rigid_body_data.device
            self._scene._gpu_fetch_all()
            # TODO (stao): unknown what happens when we do reconfigure more than once when GPU is one. figure this out
        self.agent.initialize()
        self._setup_sensors()  # for GPU sim, we have to setup sensors after we call setup gpu in order to enable loading mounted sensors
        if self._viewer is not None:
            self._setup_viewer()

    def _load_actors(self):
        """Loads all actors into the scene. Called by `self.reconfigure`"""

    def _load_articulations(self):
        """Loads all articulations into the scene. Called by `self.reconfigure`"""

    # TODO (stao): refactor this into sensor API
    def _setup_sensors(self):
        """Setup cameras in the scene. Called by `self.reconfigure`"""
        self._sensors = OrderedDict()

        for uid, camera_cfg in self._sensor_cfgs.items():
            if uid in self._agent_camera_cfgs:
                articulation = self.agent.robot
            else:
                articulation = None
            if isinstance(camera_cfg, StereoDepthCameraConfig):
                cam_cls = StereoDepthCamera
            else:
                cam_cls = Camera
            self._sensors[uid] = cam_cls(
                camera_cfg,
                self._scene,
                self._renderer_type,
                articulation=articulation,
            )

        # Cameras for rendering only
        self._render_cameras = OrderedDict()
        if self._renderer_type != "client":
            for uid, camera_cfg in self._render_camera_cfgs.items():
                self._render_cameras[uid] = Camera(
                    camera_cfg, self._scene, self._renderer_type
                )

        self._scene.sensors = self._sensors
        self._scene.render_cameras = self._render_cameras

    def _setup_lighting(self):
        # TODO (stao): remove this code out. refactor it to be inside scene builders
        """Setup lighting in the scene. Called by `self.reconfigure`"""

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
        if options is None:
            options = dict()

        # when giving a specific seed, we always set the main RNG based on that seed. This then deterministically changes the **sequence** of RNG
        # used for each episode after each call to reset with seed=none. By default this sequence of rng starts with the default main seed used which is 2022,
        # which means that when creating an environment and resetting without a seed, it will always have the same sequence of RNG for each episode.
        self._set_main_rng(seed)
        self._set_episode_rng(
            seed
        )  # we first set the first episode seed to allow environments to use it to reconfigure the environment with a seed
        self._elapsed_steps = 0
        reconfigure = options.get("reconfigure", False)
        if reconfigure:
            # Reconfigure the scene if assets change
            self.reconfigure()
        else:
            self._clear_sim_state()

        # Set the episode rng again after reconfiguration to guarantee seed reproducibility
        self._set_episode_rng(self._episode_seed)

        self.initialize_episode()
        obs = self.get_obs()
        if physx.is_gpu_enabled():
            # ensure all updates to object poses and configurations are applied on GPU after task initialization
            self._scene._gpu_apply_all()
            self._scene.px.gpu_update_articulation_kinematics()
            self._scene._gpu_fetch_all()
        else:
            obs = to_numpy(unbatch(obs))
        return obs, {}

    def _set_main_rng(self, seed):
        """Set the main random generator (e.g., to generate the seed for each episode)."""
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

    def initialize_episode(self):
        # TODO (stao): should we even split these into 4 separate functions?
        """Initialize the episode, e.g., poses of entities and articulations, and robot configuration.
        No new assets are created. Task-relevant information can be initialized here, like goals.
        """
        self._initialize_actors()
        self._initialize_articulations()
        self._initialize_agent()
        self._initialize_task()

    def _initialize_actors(self):
        """Initialize the poses of actors. Called by `self.initialize_episode`"""

    def _initialize_articulations(self):
        """Initialize the (joint) poses of articulations. Called by `self.initialize_episode`"""

    def _initialize_agent(self):
        """Initialize the (joint) poses of agent(robot). Called by `self.initialize_episode`"""

    def _initialize_task(self):
        """Initialize task-relevant information, like goals. Called by `self.initialize_episode`"""

    def _clear_sim_state(self):
        """Clear simulation state (velocities)"""
        for actor in self._scene.actors.values():
            if actor.px_body_type == "static":
                continue
            actor.set_linear_velocity([0, 0, 0])
            actor.set_angular_velocity([0, 0, 0])
        for articulation in self._scene.articulations.values():
            articulation.set_qvel(np.zeros(articulation.dof))
            # articulation.set_root_velocity([0, 0, 0])
            # articulation.set_root_angular_velocity([0, 0, 0])
        if physx.is_gpu_enabled():
            self._scene._gpu_apply_all()
            self._scene._gpu_fetch_all()
            # TODO (stao): This may be an unnecessary fetch and apply. ALSO do not fetch right after apply, no guarantee the data is updated correctly

    # -------------------------------------------------------------------------- #
    # Step
    # -------------------------------------------------------------------------- #

    def step(self, action: Union[None, np.ndarray, Dict]):
        self.step_action(action)
        self._elapsed_steps += 1
        obs = self.get_obs()
        info = self.get_info(obs=obs)
        reward = self.get_reward(obs=obs, action=action, info=info)
        terminated = info["success"]
        if physx.is_gpu_enabled():
            return obs, reward, terminated, False, info
        else:
            # On CPU sim mode, we always return numpy / python primitives without any batching.
            return unbatch(
                to_numpy(obs),
                to_numpy(reward),
                to_numpy(terminated),
                False,
                to_numpy(info),
            )

    def step_action(self, action):
        set_action = False
        if action is None:  # simulation without action
            pass
        elif isinstance(action, np.ndarray) or isinstance(action, torch.Tensor):
            action = to_tensor(action)
            set_action = True
        elif isinstance(action, dict):
            if action["control_mode"] != self.agent.control_mode:
                self.agent.set_control_mode(action["control_mode"])
            action = to_tensor(action["action"])
            set_action = True
        else:
            raise TypeError(type(action))

        if set_action:
            if self.num_envs == 1 and action.shape == self.single_action_space.shape:
                action = batch(action)
            self.agent.set_action(action)
            if physx.is_gpu_enabled():
                self._scene.px.gpu_apply_articulation_target_position()
        self._before_control_step()
        for _ in range(self._sim_steps_per_control):
            self.agent.before_simulation_step()
            with sapien.profile("step_i"):
                self._scene.step()
            self._after_simulation_step()
        if physx.is_gpu_enabled():
            self._scene._gpu_fetch_all()

    def evaluate(self, **kwargs) -> dict:
        """Evaluate whether the environment is currently in a success state."""
        raise NotImplementedError

    def get_info(self, **kwargs):
        """
        Get info about the current environment state, include elapsed steps and evaluation information
        """
        info = dict(elapsed_steps=self._elapsed_steps)
        info.update(self.evaluate(**kwargs))
        return info

    def _before_control_step(self):
        pass

    def _after_simulation_step(self):
        pass

    # -------------------------------------------------------------------------- #
    # Simulation and other gym interfaces
    # -------------------------------------------------------------------------- #
    def _set_scene_config(self):
        # cpu_workers=min(os.cpu_count(), 4) # NOTE (stao): use this if we use step_start and step_finish to enable CPU workloads between physx steps.
        # NOTE (fxiang): PCM is enabled for GPU sim regardless.
        # NOTE (fxiang): smaller contact_offset is faster as less contacts are considered, but some contacts may be missed if distance changes too fast
        # NOTE (fxiang): solver iterations 15 is recommended to balance speed and accuracy. If stable grasps are necessary >= 20 is preferred.
        # NOTE (fxiang): can try using more cpu_workers as it may also make it faster if there are a lot of collisions, collision filtering is on CPU
        # NOTE (fxiang): enable_enhanced_determinism is for CPU probably. If there are 10 far apart sub scenes, this being True makes it so they do not impact each other at all
        physx.set_scene_config(
            cpu_workers=0,
            enable_pcm=True,
            solver_iterations=15,
            contact_offset=0.02,
            solver_velocity_iterations=1,
            enable_tgs=True,
        )
        # note these frictions are same as unity
        physx.set_default_material(
            dynamic_friction=0.3, static_friction=0.3, restitution=0
        )

    def _setup_scene(self):
        """Setup the simulation scene instance.
        The function should be called in reset(). Called by `self.reconfigure`"""
        self._set_scene_config()
        if sapien.physx.is_gpu_enabled():
            if self.physx_system is not None:
                raise RuntimeError(
                    "In GPU simulation, you cannot create an environment or reconfigure more than once. You must initialize a new process"
                )
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
                        scene_x * self.gpu_sim_cfgs["spacing"],
                        scene_y * self.gpu_sim_cfgs["spacing"],
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
        self._scene = ManiSkillScene(sub_scenes)
        self.physx_system.timestep = 1.0 / self._sim_freq

    def _clear(self):
        """Clear the simulation scene instance and other buffers.
        The function can be called in reset() before a new scene is created.
        Called by `self.reconfigure` and when the environment is closed/deleted
        """
        self._close_viewer()
        self.agent = None
        self._sensors = OrderedDict()
        self._render_cameras = OrderedDict()
        self._scene = None

    def close(self):
        self._clear()

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

    def get_state(self):
        """Get environment state. Override to include task information (e.g., goal)"""
        return self._scene.get_sim_state()

    def set_state(self, state: np.ndarray):
        """Set environment state. Override to include task information (e.g., goal)"""
        return self._scene.set_sim_state(state)

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

        Called by `self.reconfigure`
        """
        # TODO (stao): handle GPU parallel sim rendering code:
        if physx.is_gpu_enabled():
            self._viewer_scene_idx = 0
        # CAUTION: `set_scene` should be called after assets are loaded.
        self._viewer.set_scene(self._scene.sub_scenes[0])
        control_window: sapien.utils.viewer.control_window.ControlWindow = (
            get_obj_by_type(
                self._viewer.plugins, sapien.utils.viewer.control_window.ControlWindow
            )
        )
        control_window.show_joint_axes = False
        control_window.show_camera_linesets = False

    def render_human(self):
        if self._viewer is None:
            self._viewer = Viewer(self._renderer)
            self._setup_viewer()
            self._viewer.set_camera_pose(
                self._render_cameras["render_camera"].camera.global_pose
            )
        self._scene.update_render()

        # TODO (stao): currently in GPU mode we cannot render all sub-scenes together in the GUI yet. So we have this
        # naive solution which shows whatever scene is selected by self._viewer_scene_idx
        if physx.is_gpu_enabled() and self._scene._buffers_ready:
            # TODO (stao): This is the slow method, update objects via cpu
            for actor in self._scene.actors.values():
                if actor.px_body_type == "static":
                    continue
                i = self._viewer_scene_idx

                actor_pose = to_numpy(actor.pose.raw_pose[i])
                entity = actor._objs[self._viewer_scene_idx]
                entity.set_pose(sapien.Pose(actor_pose[:3], actor_pose[3:]))
            for articulation in self._scene.articulations.values():
                i = self._viewer_scene_idx
                wrapped_links = articulation.links
                art = articulation._objs[self._viewer_scene_idx]
                for j, link in enumerate(art.links):
                    link_pose = to_numpy(wrapped_links[j].pose.raw_pose[i])
                    comp = link.entity.find_component_by_type(
                        sapien.render.RenderBodyComponent
                    )
                    if comp is not None:
                        comp.set_pose(sapien.Pose(link_pose[:3], link_pose[3:]))
        self._viewer.render()
        return self._viewer

    def render_rgb_array(self, camera_name: str = None):
        """Render an RGB image from the specified camera."""
        self.update_render()
        images = []
        # TODO (stao): refactor this code either into ManiSkillScene class and/or merge the code, it's pretty similar?
        if physx.is_gpu_enabled():
            for name in self._scene.render_cameras.keys():
                camera_group = self._scene.camera_groups[name]
                if camera_name is not None and name != camera_name:
                    continue
                camera_group.take_picture()
                rgb = camera_group.get_picture_cuda("Color")[..., :3].clone()
                images.append(rgb)
        else:
            for name, camera in self._scene.render_cameras.items():
                if camera_name is not None and name != camera_name:
                    continue
                camera.take_picture()
                rgb = camera.get_picture("Color")[..., :3]
                images.append(rgb)
        if len(images) == 0:
            return None
        if len(images) == 1:
            return images[0]
        return tile_images(images)

    def render_cameras(self):
        """
        Renders all sensors that the agent can use and see and displays them
        """
        images = []
        self.update_render()
        self.capture_sensor_data()
        cameras_images = self.get_images()
        for camera_images in cameras_images.values():
            images.extend(observations_to_images(camera_images))
        return tile_images(images)

    def render(self):
        """
        Either opens a viewer if render_mode is "human", or returns an array that you can use to save videos
        """
        if self.render_mode is None:
            raise RuntimeError("render_mode is not set.")
        if self.render_mode == "human":
            return self.render_human()
        elif self.render_mode == "rgb_array":
            return self.render_rgb_array()
        elif self.render_mode == "cameras":
            return self.render_cameras()
        else:
            raise NotImplementedError(f"Unsupported render mode {self.render_mode}.")

    # ---------------------------------------------------------------------------- #
    # Advanced
    # ---------------------------------------------------------------------------- #
    def gen_scene_pcd(self, num_points: int = int(1e5)) -> np.ndarray:
        """Generate scene point cloud for motion planning, excluding the robot"""
        meshes = []
        articulations = self._scene.get_all_articulations()
        if self.agent is not None:
            articulations.pop(articulations.index(self.agent.robot))
        for articulation in articulations:
            articulation_mesh = merge_meshes(get_articulation_meshes(articulation))
            if articulation_mesh:
                meshes.append(articulation_mesh)

        for actor in self._scene.get_all_actors():
            actor_mesh = merge_meshes(get_component_meshes(actor))
            if actor_mesh:
                meshes.append(
                    actor_mesh.apply_transform(
                        actor.get_pose().to_transformation_matrix()
                    )
                )

        scene_mesh = merge_meshes(meshes)
        scene_pcd = scene_mesh.sample(num_points)
        return scene_pcd
