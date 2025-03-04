import json

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots.koch.koch import Koch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.digital_twins.base_env import BaseDigitalTwinEnv
from mani_skill.envs.tasks.digital_twins.utils.camera_randomization import (
    make_camera_rectangular_prism,
    noised_look_at,
)
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SceneConfig, SimConfig


class Sim4RealBaseEnv(BaseDigitalTwinEnv):
    """
    Base environment for simulating affordable robot tasks
    General, non-task-specific randomizations to aid sim2real transfer
    Randomizations include:
        - camera position
        - camera lookat position
        - camera rotation about the viewing direction
        - subscene lighting

    Provides camera alignment points on table in debug mode
    - to align with real world points on table during camera setup

    Assumes single camera/single image observations
    """

    # TODO(xhin): generalize to multiple cameras
    # TODO(xhin): allow for option of SO-100 arm once built and tested

    # sim robot
    SUPPORTED_ROBOTS = ["koch-v1.1"]
    agent: Koch

    # rgb overlay supplied by user
    rgb_overlay_paths: dict = None  # e.g. dict(base_camera="bkground.png")
    """dict mapping camera name to the file path of the greenscreening image"""
    rgb_overlay_mode: str = "background"

    # used to allign user parameters with robot for real-life replication
    robot_x_offset = 0

    ################ Robot & camera params to match in real ################
    # all points centered at robot pos: [robot_x_offset,0,0]
    # camera parameters for lookat transform, supplied by user
    robot_base_color = [0.95, 0.95, 0.95]  # RGB
    robot_motor_color = [0.05, 0.05, 0.05]  # RGB
    base_camera_pos = [robot_x_offset + 0.40, 0 + 0.265, 0.1725]
    camera_target = [robot_x_offset + 0.2, 0, 0]
    camera_fov = 53 * (np.pi / 180)

    # debug camera position points - appear if debug in rgb_overlay_mode, useful in camera alignment
    alignment_dots = [
        [robot_x_offset + 0.2, 0 + 0.1, 0],  ## close to camera
        [robot_x_offset + 0.2, 0 - 0.1, 0],  ## far from camera
        [robot_x_offset + 0.2 + 0.15, 0, 0],  ## far infront of robot
        [
            robot_x_offset + 0.2 + 0.15,
            0 + 0.1,
            0,
        ],  ## far infront of robot and close camera
    ]

    def __init__(
        self,
        *args,
        robot_uids="koch-v1.1",
        robot_init_qpos_noise=0.02,
        num_envs=1,
        robot_base_color=None,
        robot_motor_color=None,
        rgb_overlay_path=None,
        alignment_dots=None,
        base_camera_pos=None,
        camera_target=None,
        camera_fov=None,
        cam_rand_on_step=True,
        keyframe_id=None,
        enable_shadow=False,
        debug=False,
        dr=True,
        **kwargs,
    ):
        # allow user provided camera pose
        if alignment_dots is not None:
            self.alignment_dots = alignment_dots
        if base_camera_pos is not None:
            self.base_camera_pos = base_camera_pos
        if camera_target is not None:
            self.camera_target = camera_target
        if camera_fov is not None:
            self.camera_fov = camera_fov

        # allow user provided base robot colors
        if robot_base_color is not None:
            assert (
                len(robot_base_color) == 3
            ), f"expected RGB value of length 3, instead got {robot_base_color}"
            self.robot_base_color = robot_base_color
        if robot_motor_color is not None:
            assert (
                len(robot_motor_color) == 3
            ), f"expected RGB value of length 3, instead got {robot_motor_color}"
            self.robot_motor_color = robot_motor_color

        if rgb_overlay_path is not None:
            self.rgb_overlay_path = rgb_overlay_path
            self.rgb_overlay_paths = dict(base_camera=rgb_overlay_path)
        else:
            self.rgb_overlay_path = None

        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.keyframe_id = keyframe_id
        if debug:
            overlay_mode = list(self.rgb_overlay_mode)
            overlay_mode.append("debug")
            self.rgb_overlay_mode = tuple(overlay_mode)

        toggle_rand = 0.0 if ("debug" in self.rgb_overlay_mode or dr == False) else 1.0
        self.toggle_rand = toggle_rand
        ################ Task-Agnostic Randmoizations ################
        # robot color noise
        self.robot_color_noise = 0.05

        # robot pose randomizations
        self.robot_zrot_noise = (3 * np.pi / 180) * toggle_rand
        self.robot_y_noise = 0.01 * toggle_rand  # cm offset
        # camera randomizations
        self.max_camera_offset = [
            0.025 * toggle_rand,
            0.025 * toggle_rand,
            0.025 * toggle_rand,
        ]
        # max_camera_offset = [0.05*toggle_rand, 0.05*toggle_rand, 0.05*toggle_rand]
        self.camera_target_noise = 1e-6
        self.camera_view_rot_noise = 5e-3 * toggle_rand
        self.camera_fov_noise_range = (
            2.0 * (np.pi / 180) * toggle_rand
        )  # max rad offset from camera_fov to sample from

        self.cam_rand_on_step = cam_rand_on_step and self.toggle_rand

        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            enable_shadow=enable_shadow,
            **kwargs,
        )

    ## utility functions for user control ##
    # fetching all user customizable kwargs
    def fetch_user_kwargs(self):
        return dict(
            robot_base_color=self.robot_base_color,
            robot_motor_color=self.robot_motor_color,
            rgb_overlay_path=self.rgb_overlay_path,
            alignment_dots=self.alignment_dots,
            base_camera_pos=self.base_camera_pos,
            camera_target=self.camera_target,
            camera_fov=self.camera_fov,
        )

    def save_user_kwargs(self, path):
        user_kwargs = self.fetch_user_kwargs()
        with open(path, "w") as f:
            json.dump(user_kwargs, f, indent=2)

    def get_random_camera_pose(self, n=None):
        n = self.num_envs if n is None else n
        eyes = make_camera_rectangular_prism(
            n,
            scale=self.max_camera_offset,
            center=self.base_camera_pos,
            theta=0,
        )
        return noised_look_at(
            eyes,
            target=self.camera_target,
            look_at_noise=self.camera_target_noise,
            view_axis_rot_noise=self.camera_view_rot_noise,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**22, max_rigid_patch_count=2**21
            ),
            sim_freq=120,
            control_freq=15,
        )

    @property
    def _default_sensor_configs(self):
        # mount used for camera reposing instead
        # allows keyword toggling randomization per step
        camera_fov_noise = self.camera_fov_noise_range * (
            2 * torch.rand(self.num_envs) - 1
        )
        orig_width_height = np.array([640, 480])
        scaling_factor = 480 / 128
        intrinsic = np.array([[604.923, 0, 320.502], [0, 604.595, 246.317], [0, 0, 1]])
        intrinsic[0, 2] = (
            intrinsic[0, 2] - (orig_width_height[0] - orig_width_height[1]) / 2
        )
        intrinsic[1, 2] = intrinsic[1, 2]
        intrinsic[:2, :3] = intrinsic[:2, :3] / scaling_factor

        return CameraConfig(
            "base_camera",
            pose=Pose.create_from_pq(p=[0, 0, 0]),
            width=128,
            height=128,
            # intrinsic=np.array([[604.923, 0, 320.502], [0, 604.595, 246.317], [0, 0, 1]]),
            # intrinsic=intrinsic,
            # fov=np.pi*(43/180), # 42-48
            fov=[self.camera_fov + x.item() for x in camera_fov_noise],
            near=0.01,
            far=100,
            mount=self.camera_mount,
        )

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig(
            "render_camera",
            pose=self.init_cam_poses,
            width=512,
            height=512,
            fov=self.camera_fov,
            near=0.001,
            far=100,
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0.2]))

    def _load_scene(self, options: dict):
        # table
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise * self.toggle_rand
        )
        self.table_scene.build()

        # camera mount
        cam_builder = self.scene.create_actor_builder()
        cam_builder.initial_pose = sapien.Pose(p=[0.1, 0.1, 0.1])
        self.camera_mount = cam_builder.build_kinematic("camera_mount")
        self.init_cam_poses = self.get_random_camera_pose()

        # overlay table dots for camera alignment
        def make_alignment_dot(name, color, init_pose):
            builder = self.scene.create_actor_builder()
            builder.add_cylinder_visual(
                radius=0.005,
                half_length=1e-4,
                material=sapien.render.RenderMaterial(base_color=color),
            )
            builder.initial_pose = init_pose
            return builder.build_kinematic(name=name)

        self.debugs = []
        if "debug" in self.rgb_overlay_mode:
            for i, pos in enumerate(self.alignment_dots):
                dot = make_alignment_dot(
                    f"position{i}",
                    np.array([1, 1, 0, 1]),
                    sapien.Pose(p=pos, q=euler2quat(0, np.pi / 2, 0)),
                )
                self.debugs.append(dot)
            cam_target_dot = make_alignment_dot(
                f"cam_target_dot",
                np.array([0, 1, 0, 1]),
                sapien.Pose(p=self.camera_target, q=euler2quat(0, np.pi / 2, 0)),
            )
            self.debugs.append(cam_target_dot)
            print(
                "Warning: debug mode on in digital_twins/sim_tasks/affordable_baseenv.py"
            )

        # robot color randomization
        base_color = torch.normal(
            torch.tensor(self.robot_base_color, dtype=torch.float32),
            torch.ones(3) * self.robot_color_noise * self.toggle_rand,
        ).clip(0, 1).tolist() + [1]
        motor_color = torch.normal(
            torch.tensor(self.robot_motor_color, dtype=torch.float32),
            torch.ones(3) * self.robot_color_noise * self.toggle_rand,
        ).clip(0, 1).tolist() + [1]
        self.agent.set_colors(base_color=base_color, motor_color=motor_color)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            if self.keyframe_id is not None:
                self.agent.robot.set_qpos(self.agent.keyframes[self.keyframe_id].qpos)

            # robot pose randomizations
            robot_pos = self.agent.robot.pose.p[env_idx]
            robot_pos[..., 1] += torch.normal(
                torch.zeros(b), torch.ones(b) * self.robot_y_noise
            )
            offset = self.robot_zrot_noise / 2
            base_pose = np.pi / 2
            robot_quat = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                bounds=(base_pose - offset, base_pose + offset),
            )
            self.agent.robot.set_pose(Pose.create_from_pq(p=robot_pos, q=robot_quat))

            # if cam_rand_on_step is false, this will be the only place camera randomizaiton occurs
            # cam rand per subscene
            self.camera_mount.set_pose(self.get_random_camera_pose(b))

    # camera randomziation per step
    def _before_control_step(self):
        if self.cam_rand_on_step:
            self.camera_mount.set_pose(self.get_random_camera_pose())

            # manually changing poses = update gpu buffers
            if self.gpu_sim_enabled:
                self.scene.px.gpu_apply_rigid_dynamic_data()
                self.scene.px.gpu_fetch_rigid_dynamic_data()

    # TODO (xhin): neaten
    # lighting randomization per subscene
    def _load_lighting(self, options):
        shadow = self.enable_shadow
        self.scene.set_ambient_light([0.3, 0.3, 0.3])

        direction = torch.rand(self.num_envs, 3)
        direction[..., :2] = 2 * direction[..., :2] - 1
        direction[..., -1] *= -1

        color = [1, 1, 1]
        position = [0, 0, 0]
        shadow_scale = 5
        shadow_near = -10.0
        shadow_far = 10.0
        shadow_map_size = 2048

        scene_idxs = list(range(len(self.scene.sub_scenes)))
        for scene_idx in scene_idxs:
            if self.scene.parallel_in_single_scene:
                scene = self.scene.sub_scenes[0]
            else:
                scene = self.scene.sub_scenes[scene_idx]
            entity = sapien.Entity()
            entity.name = "directional_light"
            light = sapien.render.RenderDirectionalLightComponent()
            entity.add_component(light)
            light.color = color
            light.shadow = shadow
            light.shadow_near = shadow_near
            light.shadow_far = shadow_far
            light.shadow_half_size = shadow_scale
            light.shadow_map_size = shadow_map_size
            if self.scene.parallel_in_single_scene:
                light_position = position + self.scene.scene_offsets_np[scene_idx]
            else:
                light_position = position
            light.pose = sapien.Pose(
                light_position,
                sapien.math.shortest_rotation(
                    [1, 0, 0], list(direction[scene_idx].cpu().numpy())
                ),
            )
            scene.add_entity(entity)
            if self.scene.parallel_in_single_scene:
                # for directional lights adding multiple does not make much sense
                # and for parallel gui rendering setup accurate lighting does not matter as it is only
                # for demo purposes
                break
        return
