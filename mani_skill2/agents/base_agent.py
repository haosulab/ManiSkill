from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import numpy as np
import sapien.core as sapien
from gym import spaces

from mani_skill2 import DESCRIPTION_DIR
from .base_controller import BaseController, CombinedController, ControllerConfig
from .camera import MountedCameraConfig, get_camera_images, get_camera_pcd
from mani_skill2.utils.common import merge_dicts


def parse_urdf_config(config_dict: dict, scene: sapien.Scene) -> Dict:
    """Parse config from dict for SAPIEN URDF loader.

    Args:
        config_dict (dict): a dict containing link physical properties.
        scene (sapien.Scene): simualtion scene

    Returns:
        Dict: urdf config passed to `sapien.URDFLoader.load`.
    """
    urdf_config = {}

    # Create physical materials
    materials = {}
    for k, v in config_dict.get("materials", {}).items():
        materials[k] = scene.create_physical_material(**v)

    # Specify properties for links
    link_configs = {}
    for link_name, link_config in config_dict.get("links", {}).items():
        link_config = link_config.copy()
        # substitute with actual material
        link_config["material"] = materials[link_config["material"]]
        link_configs[link_name] = link_config
    if link_configs:
        urdf_config["link"] = link_configs

    return urdf_config


@dataclass
class AgentConfig:
    """Agent configuration.

    Args:
        urdf_path: path to URDF file. Support placeholders like {description}.
        urdf_config: a dict to specify materials and simulation parameters when loading URDF from SAPIEN.
        controllers: a dict of controller configurations
        cameras: a dict of onboard camera configurations
    """

    urdf_path: str
    urdf_config: dict
    controllers: Dict[str, Union[ControllerConfig, Dict[str, ControllerConfig]]]
    cameras: Dict[str, MountedCameraConfig]


class BaseAgent:
    """Base class for agents.

    Agent is an interface of the robot (sapien.Articulation).

    Args:
        scene (sapien.Scene): simulation scene instance.
        control_freq (int): control frequency (Hz).
        control_mode: uuid of controller to use
        fix_root_link: whether to fix the robot root link
        config: agent configuration
    """

    robot: sapien.Articulation
    controllers: Dict[str, BaseController]
    cameras: Dict[str, sapien.CameraEntity]

    def __init__(
        self,
        scene: sapien.Scene,
        control_freq: int,
        control_mode: str = None,
        fix_root_link=True,
        config: AgentConfig = None,
    ):
        self.scene = scene
        self._control_freq = control_freq

        self._config = config or self.get_default_config()

        # URDF
        self.urdf_path = self._config.urdf_path
        self.fix_root_link = fix_root_link
        self.urdf_config = self._config.urdf_config

        # Controller
        self.controller_configs = self._config.controllers
        self.supported_control_modes = list(self.controller_configs.keys())
        if control_mode is None:
            control_mode = self.supported_control_modes[0]
        # The control mode after reset for consistency
        self._default_control_mode = control_mode

        # Sensors
        self.camera_configs = self._config.cameras

        self._load_articulation()
        self._setup_controllers()
        self.set_control_mode(control_mode)
        self._setup_cameras()
        self._after_init()

    def get_default_config(self) -> AgentConfig:
        raise NotImplementedError

    def _load_articulation(self):
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = self.fix_root_link

        urdf_path = str(self.urdf_path)
        urdf_path = urdf_path.format(description=DESCRIPTION_DIR)

        urdf_config = parse_urdf_config(self.urdf_config, self.scene)

        # TODO(jigu): support loading multiple convex collision shapes
        self.robot = loader.load(str(urdf_path), urdf_config)
        assert self.robot is not None, f"Fail to load URDF from {urdf_path}"
        self.robot.set_name(Path(urdf_path).stem)

    def _setup_controllers(self):
        self.controllers = OrderedDict()
        for uuid, config in self.controller_configs.items():
            if isinstance(config, dict):
                self.controllers[uuid] = CombinedController(
                    config, self.robot, self._control_freq
                )
            else:
                self.controllers[uuid] = config.controller_cls(
                    config, self.robot, self._control_freq
                )

    def _setup_cameras(self):
        self.cameras = OrderedDict()
        for uuid, config in self.camera_configs.items():
            self.cameras[uuid] = config.build(self.robot, self.scene, name=uuid)

    def _after_init(self):
        """After initialization. E.g., caching the end-effector link."""
        pass

    @property
    def control_mode(self):
        """Get the currently activated controller uuid."""
        return self._control_mode

    def set_control_mode(self, control_mode):
        """Set the controller and reset."""
        assert (
            control_mode in self.supported_control_modes
        ), "{} not in supported modes: {}".format(
            control_mode, self.supported_control_modes
        )
        self._control_mode = control_mode
        self.controller.reset()

    @property
    def controller(self):
        """Get currently activated controller."""
        if self._control_mode is None:
            raise RuntimeError("Please specify a control mode first")
        else:
            return self.controllers[self._control_mode]

    @property
    def action_space(self):
        if self._control_mode is None:
            return spaces.Dict(
                {
                    uuid: controller.action_space
                    for uuid, controller in self.controllers.items()
                }
            )
        else:
            return self.controller.action_space

    def reset(self, init_qpos=None):
        if init_qpos is not None:
            self.robot.set_qpos(init_qpos)
        self.robot.set_qvel(np.zeros(self.robot.dof))
        self.robot.set_qacc(np.zeros(self.robot.dof))
        self.robot.set_qf(np.zeros(self.robot.dof))
        self.set_control_mode(self._default_control_mode)

    def set_action(self, action):
        self.controller.set_action(action)

    def before_simulation_step(self):
        self.controller.before_simulation_step()

    # -------------------------------------------------------------------------- #
    # Observations
    # -------------------------------------------------------------------------- #
    def get_proprioception(self):
        return OrderedDict(
            qpos=self.robot.get_qpos(),
            qvel=self.robot.get_qvel(),
            # TODO(jigu): Shall we allow an empty dict here?
            controller=self.controller.get_state(),
        )

    def get_state(self) -> Dict:
        """Get current state for MPC, including robot state and controller state"""
        state = OrderedDict()

        # robot state
        root_link = self.robot.get_links()[0]
        state["robot_root_pose"] = root_link.get_pose()
        state["robot_root_vel"] = root_link.get_velocity()
        state["robot_root_qvel"] = root_link.get_angular_velocity()
        state["robot_qpos"] = self.robot.get_qpos()
        state["robot_qvel"] = self.robot.get_qvel()
        state["robot_qacc"] = self.robot.get_qacc()

        # controller state
        state["controller"] = self.controller.get_state()

        return state

    def set_state(self, state: Dict, ignore_controller=False):
        # robot state
        self.robot.set_root_pose(state["robot_root_pose"])
        self.robot.set_root_velocity(state["robot_root_vel"])
        self.robot.set_root_angular_velocity(state["robot_root_qvel"])
        self.robot.set_qpos(state["robot_qpos"])
        self.robot.set_qvel(state["robot_qvel"])
        self.robot.set_qacc(state["robot_qacc"])

        if not ignore_controller and "controller" in state:
            self.controller.set_state(state["controller"])

    def take_picture(self):
        # NOTE(jigu): take_picture, which starts rendering pipelines, is non-blocking.
        # Thus, calling it before other computation is more efficient.
        for cam in self.cameras.values():
            cam.take_picture()

    def get_camera_images(
        self, rgb=True, depth=False, visual_seg=False, actor_seg=False
    ) -> Dict[str, Dict[str, np.ndarray]]:
        # Assume scene.update_render() and camera.take_picture() are called
        ret = OrderedDict()
        base2world = self.robot.pose.to_transformation_matrix()

        for name, cam in self.cameras.items():
            images = get_camera_images(
                cam, rgb=rgb, depth=depth, visual_seg=visual_seg, actor_seg=actor_seg
            )
            images["camera_intrinsic"] = cam.get_intrinsic_matrix()
            images["camera_extrinsic"] = cam.get_extrinsic_matrix()
            images["camera_extrinsic_base_frame"] = (
                images["camera_extrinsic"] @ base2world
            )
            ret[name] = images

        return ret

    def get_camera_poses(self) -> Dict[str, np.ndarray]:
        poses = OrderedDict()
        for name, cam in self.cameras.items():
            poses[name] = cam.get_pose().to_transformation_matrix()
        return poses

    def get_camera_pcd(self, rgb=True, visual_seg=False, actor_seg=False, fuse=False):
        # Assume scene.update_render() and camera.take_picture() are called
        ret = OrderedDict()

        for name, cam in self.cameras.items():
            pcd = get_camera_pcd(cam, rgb, visual_seg, actor_seg)  # dict
            # Model matrix is the transformation from OpenGL camera space to SAPIEN world space
            # camera.get_model_matrix() must be called after scene.update_render()!
            T = cam.get_model_matrix()
            pcd["xyz"] = pcd["xyz"] @ T[:3, :3].T + T[:3, 3]
            ret[name] = pcd

        if fuse:
            return merge_dicts(ret.values())
        else:
            return ret
