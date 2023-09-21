import os
from collections import OrderedDict
from typing import Dict, Optional, Sequence, Union

import gymnasium as gym
import numpy as np
import sapien.core as sapien
from sapien.utils import Viewer

from mani_skill2 import ASSET_DIR, logger
from mani_skill2.agents.base_agent import AgentConfig, BaseAgent
from mani_skill2.sensors.camera import (
    Camera,
    CameraConfig,
    parse_camera_cfgs,
    update_camera_cfgs_from_dict,
)
from mani_skill2.sensors.depth_camera import StereoDepthCamera, StereoDepthCameraConfig
from mani_skill2.utils.common import convert_observation_to_space, flatten_state_dict
from mani_skill2.utils.sapien_utils import (
    get_actor_state,
    get_articulation_state,
    set_actor_state,
    set_articulation_state,
)
from mani_skill2.utils.trimesh_utils import (
    get_actor_meshes,
    get_articulation_meshes,
    merge_meshes,
)
from mani_skill2.utils.visualization.misc import observations_to_images, tile_images


class BaseEnv(gym.Env):
    """Superclass for ManiSkill environments.

    Args:
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
        shader_dir (str): shader directory. Defaults to "ibl".
            "ibl" and "rt" are built-in options with SAPIEN. Other options are user-defined.
        render_config (dict): kwargs to configure the renderer. Only for `SapienRenderer`.
            See `sapien.RenderConfig` for more details.
        enable_shadow (bool): whether to enable shadow for lights. Defaults to False.
        camera_cfgs (dict): configurations of cameras. See notes for more details.
        render_camera_cfgs (dict): configurations of rendering cameras. Similar usage as @camera_cfgs.

    Note:
        `camera_cfgs` is used to update environement-specific camera configurations.
        If the key is one of camera names, the value will be applied to the corresponding camera.
        Otherwise, the value will be applied to all cameras (but overridden by camera-specific values).
    """

    # fmt: off
    SUPPORTED_OBS_MODES = ("state", "state_dict", "none", "image")
    SUPPORTED_REWARD_MODES = ("normalized_dense", "dense", "sparse")
    SUPPORTED_RENDER_MODES = ("human", "rgb_array", "cameras")
    # fmt: on

    metadata = {"render_modes": SUPPORTED_RENDER_MODES}

    agent: BaseAgent
    _agent_cfg: AgentConfig
    _cameras: Dict[str, Camera]
    _camera_cfgs: Dict[str, CameraConfig]
    _agent_camera_cfgs: Dict[str, CameraConfig]
    _render_cameras: Dict[str, Camera]
    _render_camera_cfgs: Dict[str, CameraConfig]

    def __init__(
        self,
        obs_mode: str = None,
        reward_mode: str = None,
        control_mode: str = None,
        render_mode: str = None,
        sim_freq: int = 500,
        control_freq: int = 20,
        renderer: str = "sapien",
        renderer_kwargs: dict = None,
        shader_dir: str = "ibl",
        render_config: dict = None,
        enable_shadow: bool = False,
        camera_cfgs: dict = None,
        render_camera_cfgs: dict = None,
        bg_name: str = None,
    ):
        # Create SAPIEN engine
        self._engine = sapien.Engine()
        # TODO(jigu): Change to `warning` after lighting in VecEnv is fixed.
        self._engine.set_log_level(os.getenv("MS2_SIM_LOG_LEVEL", "error"))

        # Create SAPIEN renderer
        self._renderer_type = renderer
        if renderer_kwargs is None:
            renderer_kwargs = {}
        if self._renderer_type == "sapien":
            self._renderer = sapien.SapienRenderer(**renderer_kwargs)
            if shader_dir == "ibl":
                _render_config = dict(camera_shader_dir="ibl", viewer_shader_dir="ibl")
            elif shader_dir == "rt":
                _render_config = dict(
                    camera_shader_dir="rt",
                    viewer_shader_dir="rt",
                    rt_samples_per_pixel=32,
                    rt_max_path_depth=8,
                    rt_use_denoiser=True,
                )
            else:
                _render_config = dict(
                    camera_shader_dir=shader_dir, viewer_shader_dir=shader_dir
                )
            if render_config is None:
                render_config = {}
            _render_config.update(render_config)
            for k, v in _render_config.items():
                setattr(sapien.render_config, k, v)
            self._renderer.set_log_level(os.getenv("MS2_RENDERER_LOG_LEVEL", "warn"))
        elif self._renderer_type == "client":
            self._renderer = sapien.RenderClient(**renderer_kwargs)
            # TODO(jigu): add `set_log_level` for RenderClient?
        else:
            raise NotImplementedError(self._renderer_type)

        self._engine.set_renderer(self._renderer)

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

        # NOTE(jigu): Agent and camera configurations should not change after initialization.
        self._configure_agent()
        self._configure_cameras()
        self._configure_render_cameras()
        # Override camera configurations
        if camera_cfgs is not None:
            update_camera_cfgs_from_dict(self._camera_cfgs, camera_cfgs)
        if render_camera_cfgs is not None:
            update_camera_cfgs_from_dict(self._render_camera_cfgs, render_camera_cfgs)

        # Lighting
        self.enable_shadow = enable_shadow

        # Visual background
        self.bg_name = bg_name

        # Use a fixed (main) seed to enhance determinism
        self._main_seed = None
        self.set_main_rng(2022)
        obs, _ = self.reset(seed=2022, options=dict(reconfigure=True))
        self.observation_space = convert_observation_to_space(obs)
        if self._obs_mode == "image":
            image_obs_space = self.observation_space.spaces["image"]
            for uid, camera in self._cameras.items():
                image_obs_space.spaces[uid] = camera.observation_space
        self.action_space = self.agent.action_space

    def _configure_agent(self):
        # TODO(jigu): Support a dummy agent for simulation only
        raise NotImplementedError

    def _configure_cameras(self):
        self._camera_cfgs = OrderedDict()
        self._camera_cfgs.update(parse_camera_cfgs(self._register_cameras()))

        self._agent_camera_cfgs = OrderedDict()
        if self._agent_cfg is not None:
            self._agent_camera_cfgs = parse_camera_cfgs(self._agent_cfg.cameras)
            self._camera_cfgs.update(self._agent_camera_cfgs)

    def _register_cameras(
        self,
    ) -> Union[CameraConfig, Sequence[CameraConfig], Dict[str, CameraConfig]]:
        """Register (non-agent) cameras for the environment."""
        return []

    def _configure_render_cameras(self):
        self._render_camera_cfgs = parse_camera_cfgs(self._register_render_cameras())

    def _register_render_cameras(
        self,
    ) -> Union[CameraConfig, Sequence[CameraConfig], Dict[str, CameraConfig]]:
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
        if self._obs_mode == "none":
            # Some cases do not need observations, e.g., MPC
            return OrderedDict()
        elif self._obs_mode == "state":
            state_dict = self._get_obs_state_dict()
            return flatten_state_dict(state_dict)
        elif self._obs_mode == "state_dict":
            return self._get_obs_state_dict()
        elif self._obs_mode == "image":
            return self._get_obs_images()
        else:
            raise NotImplementedError(self._obs_mode)

    def _get_obs_state_dict(self):
        """Get (GT) state-based observations."""
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
        self._scene.update_render()

    def take_picture(self):
        """Take pictures from all cameras (non-blocking)."""
        for cam in self._cameras.values():
            cam.take_picture()

    def get_images(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Get (raw) images from all cameras (blocking)."""
        images = OrderedDict()
        for name, cam in self._cameras.items():
            images[name] = cam.get_images()
        return images

    def get_camera_params(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Get camera parameters from all cameras."""
        params = OrderedDict()
        for name, cam in self._cameras.items():
            params[name] = cam.get_params()
        return params

    def _get_obs_images(self) -> OrderedDict:
        if self._renderer_type == "client":
            # NOTE: not compatible with StereoDepthCamera
            cameras = [x.camera for x in self._cameras.values()]
            self._scene._update_render_and_take_pictures(cameras)
        else:
            self.update_render()
            self.take_picture()
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

    def get_reward(self, **kwargs):
        if self._reward_mode == "sparse":
            eval_info = self.evaluate(**kwargs)
            return float(eval_info["success"])
        elif self._reward_mode == "dense":
            return self.compute_dense_reward(**kwargs)
        elif self._reward_mode == "normalized_dense":
            return self.compute_normalized_dense_reward(**kwargs)
        else:
            raise NotImplementedError(self._reward_mode)

    def compute_dense_reward(self, **kwargs):
        raise NotImplementedError

    def compute_normalized_dense_reward(self, **kwargs):
        raise NotImplementedError

    # -------------------------------------------------------------------------- #
    # Reconfigure
    # -------------------------------------------------------------------------- #
    def reconfigure(self):
        """Reconfigure the simulation scene instance.
        This function should clear the previous scene, and create a new one.
        """
        self._clear()

        self._setup_scene()
        self._load_agent()
        self._load_actors()
        self._load_articulations()
        self._setup_cameras()
        self._setup_lighting()

        # Cache actors and articulations
        self._actors = self.get_actors()
        self._articulations = self.get_articulations()

        self._load_background()

        if self._viewer is not None:
            self._setup_viewer()

    def _add_ground(self, altitude=0.0, render=True):
        if render:
            rend_mtl = self._renderer.create_material()
            rend_mtl.base_color = [0.06, 0.08, 0.12, 1]
            rend_mtl.metallic = 0.0
            rend_mtl.roughness = 0.9
            rend_mtl.specular = 0.8
        else:
            rend_mtl = None
        return self._scene.add_ground(
            altitude=altitude,
            render=render,
            render_material=rend_mtl,
        )

    def _load_actors(self):
        pass

    def _load_articulations(self):
        pass

    def _load_agent(self):
        pass

    def _setup_cameras(self):
        self._cameras = OrderedDict()
        for uid, camera_cfg in self._camera_cfgs.items():
            if uid in self._agent_camera_cfgs:
                articulation = self.agent.robot
            else:
                articulation = None
            if isinstance(camera_cfg, StereoDepthCameraConfig):
                cam_cls = StereoDepthCamera
            else:
                cam_cls = Camera
            self._cameras[uid] = cam_cls(
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

    def _setup_lighting(self):
        if self.bg_name is not None:
            return

        shadow = self.enable_shadow
        self._scene.set_ambient_light([0.3, 0.3, 0.3])
        # Only the first of directional lights can have shadow
        self._scene.add_directional_light(
            [1, 1, -1], [1, 1, 1], shadow=shadow, scale=5, shadow_map_size=2048
        )
        self._scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _load_background(self):
        if self.bg_name is None:
            return

        # Remove all existing lights
        for l in self._scene.get_all_lights():
            self._scene.remove_light(l)

        if self.bg_name == "minimal_bedroom":
            # "Minimalistic Modern Bedroom" (https://skfb.ly/oCnNx) by dylanheyes is licensed under Creative Commons Attribution (http://creativecommons.org/licenses/by/4.0/).
            path = ASSET_DIR / "background/minimalistic_modern_bedroom.glb"
            pose = sapien.Pose([0, 0, 1.7], [0.5, 0.5, -0.5, -0.5])
            self._scene.set_ambient_light([0.1, 0.1, 0.1])
            self._scene.add_point_light([-0.349, 0, 1.4], [1.0, 0.9, 0.9])
        else:
            raise NotImplementedError("Unsupported background: {}".format(self.bg_name))

        if not path.exists():
            raise FileNotFoundError(
                f"The visual background asset is not found: {path}."
                "Please download the background asset by `python -m mani_skill2.utils.download_asset {}`".format(
                    self.bg_name
                )
            )

        builder = self._scene.create_actor_builder()
        builder.add_visual_from_file(str(path))
        self.visual_bg = builder.build_kinematic()
        self.visual_bg.set_pose(pose)

    # -------------------------------------------------------------------------- #
    # Reset
    # -------------------------------------------------------------------------- #
    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()

        # when giving a specific seed, we always set the main RNG based on that seed. This then deterministically changes the **sequence** of RNG 
        # used for each episode after each call to reset with seed=none. By default this sequence of rng starts with the default main seed used which is 2022,
        # which means that when creating an environment and resetting without a seed, it will always have the same sequence of RNG for each episode.
        self.set_main_rng(seed)
        self.set_episode_rng(seed) # we first set the first episode seed to allow environments to use it to reconfigure the environment with a seed
        self._elapsed_steps = 0
        reconfigure = options.get("reconfigure", False)
        if reconfigure:
            # Reconfigure the scene if assets change
            self.reconfigure()
        else:
            self._clear_sim_state()

        # Set the episode rng again after reconfiguration to guarantee seed reproducibility
        self.set_episode_rng(self._episode_seed)
        self.initialize_episode()

        return self.get_obs(), {}

    def set_main_rng(self, seed):
        """Set the main random generator (e.g., to generate the seed for each episode)."""
        if seed is None:
            if self._main_seed is not None:
                return
            seed = np.random.RandomState().randint(2**32)
        self._main_seed = seed
        self._main_rng = np.random.RandomState(self._main_seed)

    def set_episode_rng(self, seed):
        """Set the random generator for current episode."""
        if seed is None:
            self._episode_seed = self._main_rng.randint(2**32)
        else:
            self._episode_seed = seed
        self._episode_rng = np.random.RandomState(self._episode_seed)

    def initialize_episode(self):
        """Initialize the episode, e.g., poses of actors and articulations, and robot configuration.
        No new assets are created. Task-relevant information can be initialized here, like goals.
        """
        self._initialize_actors()
        self._initialize_articulations()
        self._initialize_agent()
        self._initialize_task()

    def _initialize_actors(self):
        """Initialize the poses of actors."""
        pass

    def _initialize_articulations(self):
        """Initialize the (joint) poses of articulations."""
        pass

    def _initialize_agent(self):
        """Initialize the (joint) poses of agent(robot)."""
        pass

    def _initialize_task(self):
        """Initialize task-relevant information, like goals."""
        pass

    def _clear_sim_state(self):
        """Clear simulation state (velocities)"""
        for actor in self._scene.get_all_actors():
            if actor.type != "static":
                # TODO(fxiang): kinematic actor may need another way.
                actor.set_velocity([0, 0, 0])
                actor.set_angular_velocity([0, 0, 0])
        for articulation in self._scene.get_all_articulations():
            articulation.set_qvel(np.zeros(articulation.dof))
            articulation.set_root_velocity([0, 0, 0])
            articulation.set_root_angular_velocity([0, 0, 0])

    # -------------------------------------------------------------------------- #
    # Step
    # -------------------------------------------------------------------------- #
    def step(self, action: Union[None, np.ndarray, Dict]):
        self.step_action(action)
        self._elapsed_steps += 1

        obs = self.get_obs()
        info = self.get_info(obs=obs)
        reward = self.get_reward(obs=obs, action=action, info=info)
        terminated = self.get_done(obs=obs, info=info)
        return obs, reward, terminated, False, info

    def step_action(self, action):
        if action is None:  # simulation without action
            pass
        elif isinstance(action, np.ndarray):
            self.agent.set_action(action)
        elif isinstance(action, dict):
            if action["control_mode"] != self.agent.control_mode:
                self.agent.set_control_mode(action["control_mode"])
            self.agent.set_action(action["action"])
        else:
            raise TypeError(type(action))

        self._before_control_step()
        for _ in range(self._sim_steps_per_control):
            self.agent.before_simulation_step()
            self._scene.step()
            self._after_simulation_step()

    def evaluate(self, **kwargs) -> dict:
        """Evaluate whether the task succeeds."""
        raise NotImplementedError

    def get_done(self, info: dict, **kwargs):
        return bool(info["success"])

    def get_info(self, **kwargs):
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
    def _get_default_scene_config(self):
        scene_config = sapien.SceneConfig()
        scene_config.default_dynamic_friction = 1.0
        scene_config.default_static_friction = 1.0
        scene_config.default_restitution = 0.0
        scene_config.contact_offset = 0.02
        scene_config.enable_pcm = False
        scene_config.solver_iterations = 25
        # NOTE(fanbo): solver_velocity_iterations=0 is undefined in PhysX
        scene_config.solver_velocity_iterations = 1
        if self._renderer_type == "client":
            scene_config.disable_collision_visual = True
        return scene_config

    def _setup_scene(self, scene_config: Optional[sapien.SceneConfig] = None):
        """Setup the simulation scene instance.
        The function should be called in reset().
        """
        if scene_config is None:
            scene_config = self._get_default_scene_config()
        self._scene = self._engine.create_scene(scene_config)
        self._scene.set_timestep(1.0 / self._sim_freq)

    def _clear(self):
        """Clear the simulation scene instance and other buffers.
        The function can be called in reset() before a new scene is created.
        """
        self._close_viewer()
        self.agent = None
        self._cameras = OrderedDict()
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
    def get_actors(self):
        return self._scene.get_all_actors()

    def get_articulations(self):
        articulations = self._scene.get_all_articulations()
        # NOTE(jigu): There might be dummy articulations used by controllers.
        # TODO(jigu): Remove dummy articulations if exist.
        return articulations

    def get_sim_state(self) -> np.ndarray:
        """Get simulation state."""
        state = []
        for actor in self._actors:
            state.append(get_actor_state(actor))
        for articulation in self._articulations:
            state.append(get_articulation_state(articulation))
        return np.hstack(state)

    def set_sim_state(self, state: np.ndarray):
        """Set simulation state."""
        KINEMANTIC_DIM = 13  # [pos, quat, lin_vel, ang_vel]
        start = 0
        for actor in self._actors:
            set_actor_state(actor, state[start : start + KINEMANTIC_DIM])
            start += KINEMANTIC_DIM
        for articulation in self._articulations:
            ndim = KINEMANTIC_DIM + 2 * articulation.dof
            set_articulation_state(articulation, state[start : start + ndim])
            start += ndim

    def get_state(self):
        """Get environment state. Override to include task information (e.g., goal)"""
        return self.get_sim_state()

    def set_state(self, state: np.ndarray):
        """Set environment state. Override to include task information (e.g., goal)"""
        return self.set_sim_state(state)

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
        """
        # CAUTION: `set_scene` should be called after assets are loaded.
        self._viewer.set_scene(self._scene)
        self._viewer.toggle_axes(False)
        self._viewer.toggle_camera_lines(False)

    def render_human(self):
        self.update_render()
        if self._viewer is None:
            self._viewer = Viewer(self._renderer)
            self._setup_viewer()
        self._viewer.render()
        return self._viewer

    def render_rgb_array(self, camera_name: str = None):
        """Render an RGB image from the specified camera."""
        self.update_render()
        images = []
        for name, camera in self._render_cameras.items():
            if camera_name is not None and name != camera_name:
                continue
            rgba = camera.get_images(take_picture=True)["Color"]
            rgb = np.uint8(np.clip(rgba[..., :3], 0, 1) * 255)
            images.append(rgb)
        if len(images) == 0:
            return None
        if len(images) == 1:
            return images[0]
        return tile_images(images)

    def render_cameras(self):
        images = []
        self.render_mode = "rgb_array"
        rgb_array = self.render()
        self.render_mode = "cameras"
        if rgb_array is not None:
            images.append(rgb_array)
        images.extend(self._render_cameras_images())
        return tile_images(images)

    def _render_cameras_images(self):
        images = []
        self.update_render()
        self.take_picture()
        cameras_images = self.get_images()
        for camera_images in cameras_images.values():
            images.extend(observations_to_images(camera_images))
        return images

    def render(self):
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
            actor_mesh = merge_meshes(get_actor_meshes(actor))
            if actor_mesh:
                meshes.append(
                    actor_mesh.apply_transform(
                        actor.get_pose().to_transformation_matrix()
                    )
                )

        scene_mesh = merge_meshes(meshes)
        scene_pcd = scene_mesh.sample(num_points)
        return scene_pcd
