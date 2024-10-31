import os
import numpy as np
import sapien
import torch
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers.passive_controller import PassiveControllerConfig
from mani_skill.agents.controllers.pd_joint_pos import PDJointPosControllerConfig
from mani_skill.envs.tasks.control.cartpole import CartpoleBalanceEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SceneConfig, SimConfig
from typing import Optional, Union
MJCF_FILE = f"{os.path.join(os.path.dirname(__file__), 'assets/cartpole.xml')}"


class CartPoleRobot(BaseAgent):
    uid = "cart_pole"
    mjcf_path = MJCF_FILE
    disable_self_collisions = True

    @property
    def _controller_configs(self):
        # NOTE it is impossible to copy joint properties from original xml files, have to tune manually until
        # it looks approximately correct
        pd_joint_delta_pos = PDJointPosControllerConfig(
            ["slider"],
            -1,
            1,
            damping=200,
            stiffness=2000,
            use_delta=True,
        )
        rest = PassiveControllerConfig(["hinge_1"], damping=0, friction=0)
        return dict(
            pd_joint_delta_pos=dict(
                slider=pd_joint_delta_pos, rest=rest, balance_passive_force=False
            )
        )

    def _load_articulation(self, initial_pose: Optional[Union[sapien.Pose, Pose]] = None):
        """
        Load the robot articulation
        """
        loader = self.scene.create_mjcf_loader()
        asset_path = str(self.mjcf_path)

        loader.name = self.uid

        # only need the robot
        builder = loader.parse(asset_path)["articulation_builders"][0]
        builder.initial_pose = initial_pose
        self.robot = builder.build(name="cartpole")
        assert self.robot is not None, f"Fail to load URDF/MJCF from {asset_path}"

        # Cache robot link ids
        self.robot_link_names = [link.name for link in self.robot.get_links()]


@register_env("CartpoleBalanceBenchmark-v1", max_episode_steps=1000)
class CartPoleBalanceBenchmarkEnv(CartpoleBalanceEnv):
    def __init__(
        self, *args, camera_width=128, camera_height=128, num_cameras=1, **kwargs
    ):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.num_cameras = num_cameras
        super().__init__(*args, robot_uids=CartPoleRobot, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=120,
            spacing=20,
            control_freq=60,
            scene_config=SceneConfig(
                bounce_threshold=0.5,
                solver_position_iterations=4,
                solver_velocity_iterations=0,
            ),
        )

    @property
    def _default_sensor_configs(self):
        from transforms3d.euler import euler2quat

        q = euler2quat(0, np.deg2rad(11.988), np.pi / 2)
        pose = sapien.Pose((0.0, -4.0, 3.0), q=q)
        sensor_configs = []
        if self.num_cameras is not None:
            for i in range(self.num_cameras):
                sensor_configs.append(
                    CameraConfig(
                        uid=f"base_camera_{i}",
                        pose=pose,
                        width=self.camera_width,
                        height=self.camera_height,
                        far=25,
                        fov=0.63,
                    )
                )
        return sensor_configs

    @property
    def _default_human_render_camera_configs(self):
        return dict()

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose())

    def _load_scene(self, options: dict):
        loader = self.scene.create_mjcf_loader()
        actor_builders = loader.parse(MJCF_FILE)["actor_builders"]
        for a in actor_builders:
            a.initial_pose = sapien.Pose()
            a.build(a.name)
        # isaac uses a 0.5mx0.5m grid so we downscale the grid which is 4x4 squares by 2 by assumign the texture square length is 2
        self.ground = build_ground(
            self.scene,
            texture_file=os.path.join(
                os.path.dirname(__file__), "assets/black_grid.png"
            ),
            texture_square_len=2,
            mipmap_levels=7,
        )

    def _load_lighting(self, options: dict):
        """Loads lighting into the scene. Called by `self._reconfigure`. If not overriden will set some simple default lighting"""
        self.scene.set_ambient_light(np.array([1, 1, 1]) * 0.3)
        for i in range(self.num_envs):
            self.scene.sub_scenes[i].set_environment_map(
                os.path.join(
                    os.path.dirname(__file__), "kloofendal_28d_misty_puresky_1k.hdr"
                )
            )

    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs, device=self.device)
