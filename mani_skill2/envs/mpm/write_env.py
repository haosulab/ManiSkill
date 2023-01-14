from collections import OrderedDict

import h5py
import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat

from mani_skill2 import ASSET_DIR
from mani_skill2.agents.configs.panda.variants import PandaStickConfig
from mani_skill2.agents.robots.panda import Panda
from mani_skill2.envs.mpm.base_env import MPMBaseEnv, MPMModelBuilder, MPMSimulator
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import vectorize_pose
from mani_skill2.sensors.camera import CameraConfig

from mani_skill2.envs.mpm.utils import load_h5_as_dict

import warp as wp
from mpm.height_rasterizer import rasterize_clear_kernel, rasterize_kernel


@wp.kernel
def success_iou_kernel(
    goal_image: wp.array(dtype=int, ndim=2),
    current_image: wp.array(dtype=int, ndim=2),
    out: wp.array(dtype=int),
):
    h, w = wp.tid()
    a = goal_image[h, w] < 40
    an = goal_image[h, w] < 50
    b = current_image[h, w] < 40
    bn = current_image[h, w] < 50

    # intersection
    if (an and b) or (a and bn):
        wp.atomic_add(out, 0, 1)

    # union
    if a or b:
        wp.atomic_add(out, 1, 1)


@register_env("Write-v0", max_episode_steps=200)
class WriteEnv(MPMBaseEnv):
    def __init__(
        self,
        *args,
        level_dir="write/levels",
        **kwargs,
    ):
        self.level_dir = ASSET_DIR / level_dir
        self.all_filepaths = sorted(self.level_dir.glob("*.h5"))
        if len(self.all_filepaths) == 0:
            raise RuntimeError(
                "Please download required assets for Write:"
                "`python -m mani_skill2.utils.download_asset write`"
            )
        super().__init__(*args, **kwargs)

    def reset(self, *args, seed=None, level_file=None, **kwargs):
        self.level_file = level_file
        return super().reset(*args, seed=seed, **kwargs)

    def _get_obs_extra(self):
        return OrderedDict(
            tcp_pose=vectorize_pose(self.end_effector.pose),
            goal=self.goal_image_display_numpy,
        )

    def _initialize_mpm(self):
        if self.level_file is not None:
            filepath = self.level_dir / self.level_file
            assert filepath is not None
        else:
            filepath = self._episode_rng.choice(self.all_filepaths)

        self.info = load_h5_as_dict(h5py.File(str(filepath), "r"))

        n = len(self.info["goal"])
        self.image_size = 64
        self.image_world_size = 0.21
        self.goal_array = wp.array(self.info["goal"], dtype=wp.vec3, device="cuda")
        self.goal_image = wp.empty(
            (self.image_size, self.image_size), dtype=int, device="cuda"
        )
        self.current_image = wp.empty(
            (self.image_size, self.image_size), dtype=int, device="cuda"
        )
        self.iou_buffer = wp.empty(2, dtype=int, device="cuda")

        wp.launch(
            rasterize_clear_kernel,
            dim=(self.image_size, self.image_size),
            inputs=[self.goal_image, 0],
            device="cuda",
        )
        xy_scale = self.image_size / self.image_world_size
        z_scale = 1000
        radius = int(0.007 * xy_scale)
        wp.launch(
            rasterize_kernel,
            dim=n,
            inputs=[
                self.goal_array,
                wp.vec3(self.image_world_size / 2, self.image_world_size / 2, 0.0),
                xy_scale,
                z_scale,
                radius,
                self.image_size,
                self.image_size,
                self.goal_image,
            ],
            device="cuda",
        )

        self.goal_image_display_numpy = self.goal_image.numpy()[:, ::-1]
        self.goal_image_display_numpy = np.clip(
            self.goal_image_display_numpy, 0, 255
        ).astype(np.uint8)

        self._episode_rng = np.random.RandomState(self._episode_seed)

        self.model_builder.clear_particles()
        E = 3e5
        nu = 0.1
        mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
        type = 0
        ys = 2e3

        height_map = np.ones((42, 42), dtype=np.float32) * 0.05
        self.model_builder.add_mpm_from_height_map(
            pos=(0.0, 0.0, 0.0),
            vel=(0.0, 0.0, 0.0),
            dx=0.005,
            height_map=height_map,
            density=3.0e3,
            mu_lambda_ys=(mu, lam, ys),
            friction_cohesion=(0.0, 0.0, 0.0),
            type=type,
            jitter=True,
            color=(0.65237011, 0.14198029, 0.02201299),
            random_state=self._episode_rng,
        )

        self.model_builder.init_model_state(self.mpm_model, self.mpm_states)
        self.mpm_model.struct.static_ke = 100.0
        self.mpm_model.struct.static_kd = 0.0
        self.mpm_model.struct.static_mu = 1.0
        self.mpm_model.struct.static_ka = 0.0

        self.mpm_model.struct.body_ke = 100.0
        self.mpm_model.struct.body_kd = 0.0
        self.mpm_model.struct.body_mu = 1.0
        self.mpm_model.struct.body_ka = 0.0

        self.mpm_model.adaptive_grid = False
        self.mpm_model.grid_contact = False
        self.mpm_model.particle_contact = True
        self.mpm_model.struct.body_sticky = 0
        self.mpm_model.struct.ground_sticky = 1

        self.mpm_model.struct.particle_radius = 0.005

    def _setup_mpm(self):
        self.model_builder = MPMModelBuilder()
        self.model_builder.set_mpm_domain(domain_size=[0.5, 0.5, 0.5], grid_length=0.01)
        self.model_builder.reserve_mpm_particles(count=self.max_particles)
        self._setup_mpm_bodies()
        self.mpm_simulator = MPMSimulator(device="cuda")
        self.mpm_model = self.model_builder.finalize(device="cuda")
        self.mpm_model.gravity = np.array((0.0, 0.0, -9.81), dtype=np.float32)
        self.mpm_model.struct.ground_normal = wp.vec3(0.0, 0.0, 1.0)
        self.mpm_model.struct.particle_radius = 0.005
        self.mpm_states = [
            self.mpm_model.state() for _ in range(self._mpm_step_per_sapien_step + 1)
        ]

    def _register_cameras(self):
        p, q = [-0.2, 0, 0.3], euler2quat(0, np.pi / 6, 0)
        return CameraConfig("base_camera", p, q, 128, 128, np.pi / 2, 0.001, 10)

    def _register_render_cameras(self):
        p, q = [-0.3, 0, 0.4], euler2quat(0, np.pi / 5, 0)
        return CameraConfig("render_camera", p, q, 512, 512, 1, 0.001, 10)

    def _configure_agent(self):
        self._agent_cfg = PandaStickConfig()

    def _load_agent(self):
        self.agent = Panda(
            self._scene,
            self._control_freq,
            control_mode=self._control_mode,
            config=self._agent_cfg,
        )
        self.end_effector = self.agent.robot.get_links()[-1]

    def _initialize_agent(self):
        noise = self._episode_rng.uniform([-0.1] * 7, [0.1] * 7)
        # fmt: off
        qpos = np.array([-0.029177314, 0.10816099, 0.03054934, -2.1639752, -0.0013982388, 2.2785723, 0.79039097]) + noise
        # fmt: on

        self.agent.reset(qpos)
        self.agent.robot.set_pose(sapien.Pose([-0.55, 0, 0]))

    def _load_actors(self):
        super()._load_actors()
        b = self._scene.create_actor_builder()
        b.add_box_collision(half_size=[0.15, 0.02, 0.04])
        b.add_box_visual(half_size=[0.15, 0.02, 0.04])
        w0 = b.build_kinematic("wall")
        w1 = b.build_kinematic("wall")
        w2 = b.build_kinematic("wall")
        w3 = b.build_kinematic("wall")

        w0.set_pose(sapien.Pose([0, -0.13, 0.04]))
        w1.set_pose(sapien.Pose([0, 0.13, 0.04]))
        w2.set_pose(sapien.Pose([-0.13, 0, 0.04], [0.7071068, 0, 0, 0.7071068]))
        w3.set_pose(sapien.Pose([0.13, 0, 0.04], [0.7071068, 0, 0, 0.7071068]))
        self.walls = [w0, w1, w2, w3]

    def _get_coupling_actors(self):
        return [
            l for l in self.agent.robot.get_links() if l.name == "panda_hand"
        ] + self.walls

    def step(self, *args, **kwargs):
        self._iou = None
        return super().step(*args, **kwargs)

    def _compute_iou(self):
        if self._iou is not None:
            return self._iou

        n = len(self.info["goal"])
        self.iou_buffer.zero_()

        wp.launch(
            rasterize_clear_kernel,
            dim=(self.image_size, self.image_size),
            inputs=[self.current_image, 0],
            device="cuda",
        )
        xy_scale = self.image_size / self.image_world_size
        z_scale = 1000
        radius = int(0.007 * xy_scale)
        wp.launch(
            rasterize_kernel,
            dim=n,
            inputs=[
                self.mpm_states[0].struct.particle_q,
                wp.vec3(self.image_world_size / 2, self.image_world_size / 2, 0.0),
                xy_scale,
                z_scale,
                radius,
                self.image_size,
                self.image_size,
                self.current_image,
            ],
            device="cuda",
        )

        wp.launch(
            success_iou_kernel,
            dim=(self.image_size, self.image_size),
            inputs=[self.goal_image, self.current_image, self.iou_buffer],
            device="cuda",
        )
        inter, union = self.iou_buffer.numpy()
        self._iou = inter / union
        return self._iou

    def evaluate(self, **kwargs):
        return OrderedDict(success=self._compute_iou() > 0.8, iou=self._compute_iou())

    def compute_dense_reward(self, reward_info=False, **kwargs):
        # iou (goal): [0, 1]
        iou = self._compute_iou()

        # reaching reward
        gripper_mat = self.end_effector.get_pose().to_transformation_matrix()
        gripper_bottom_mat = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.02], [0, 0, 0, 1]]
        )
        bottom_mat = gripper_mat @ gripper_bottom_mat
        bottom_pos = sapien.Pose.from_transformation_matrix(bottom_mat).p
        particles_x = self.get_mpm_state()["x"]
        distance = np.min(np.linalg.norm(particles_x - bottom_pos, axis=-1))
        reaching_reward = 1 - np.tanh(10.0 * distance)

        # vertical reward
        gripper_mat = self.end_effector.pose.to_transformation_matrix()
        z = gripper_mat[:3, 2]
        angle = np.arcsin(np.clip(np.linalg.norm(np.cross(z, [0, 0, -1])), -1, 1))
        reward_orientation = 1 - angle

        return iou + 0.1 * reaching_reward + 0.1 * reward_orientation
