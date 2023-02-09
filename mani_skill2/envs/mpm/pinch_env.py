from collections import OrderedDict

import h5py
import numpy as np
import sapien.core as sapien
import warp as wp
from transforms3d.euler import euler2quat
from warp.distance import compute_chamfer_distance

from mani_skill2 import ASSET_DIR
from mani_skill2.agents.configs.panda.variants import PandaPinchConfig
from mani_skill2.agents.robots.panda import Panda
from mani_skill2.envs.mpm.base_env import MPMBaseEnv, MPMModelBuilder, MPMSimulator
from mani_skill2.envs.mpm.utils import load_h5_as_dict
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import get_entity_by_name, vectorize_pose


@register_env("Pinch-v0", max_episode_steps=300)
class PinchEnv(MPMBaseEnv):
    def __init__(
        self,
        *args,
        level_dir="pinch/levels",
        **kwargs,
    ):
        self.level_dir = ASSET_DIR / level_dir
        self.all_filepaths = sorted(self.level_dir.glob("*.h5"))
        if len(self.all_filepaths) == 0:
            raise FileNotFoundError(
                "Please download required assets for Pinch:"
                "`python -m mani_skill2.utils.download_asset pinch`"
            )

        super().__init__(*args, **kwargs)

    def reset(
        self,
        *args,
        seed=None,
        level_file=None,
        **kwargs,
    ):
        self.level_file = level_file
        return super().reset(*args, seed=seed, **kwargs)

    def _setup_mpm(self):
        self.model_builder = MPMModelBuilder()
        self.model_builder.set_mpm_domain(domain_size=[0.5, 0.5, 0.5], grid_length=0.01)
        self.model_builder.reserve_mpm_particles(count=self.max_particles)

        self._setup_mpm_bodies()

        self.mpm_simulator = MPMSimulator(device="cuda")
        self.mpm_model = self.model_builder.finalize(device="cuda")
        self.mpm_model.gravity = np.array((0.0, 0.0, -1), dtype=np.float32)
        self.mpm_model.struct.ground_normal = wp.vec3(0.0, 0.0, 1.0)
        self.mpm_model.struct.particle_radius = 0.005
        self.mpm_states = [
            self.mpm_model.state() for _ in range(self._mpm_step_per_sapien_step + 1)
        ]

    def _initialize_mpm(self):
        if self.level_file is not None:
            filepath = self.level_dir / self.level_file
            assert filepath is not None
        else:
            filepath = self._episode_rng.choice(self.all_filepaths)

        # reset rng after choosing file
        self._episode_rng = np.random.RandomState(self._episode_seed)

        self.info = load_h5_as_dict(h5py.File(str(filepath), "r"))
        self.goal_depths = self.info["goal_depths"].reshape((4, 128, 128))
        self.goal_rgbs = self.info["goal_rgbs"].reshape((4, 128, 128, 3))
        self.goal_points = self.get_goal_points()

        n = len(self.info["init_state"]["mpm"]["x"])
        self.mpm_model.struct.n_particles = n
        self.set_sim_state(self.info["init_state"])
        self.mpm_model.mpm_particle_colors = (
            np.ones((n, 3)) * np.array([0.65237011, 0.14198029, 0.02201299])
        ).astype(np.float32)

        self.goal_array = wp.array(
            self.info["end_state"]["mpm"]["x"], dtype=wp.vec3, device="cuda"
        )
        self.dist1 = wp.empty(n, dtype=float, device="cuda")
        self.dist2 = wp.empty(n, dtype=float, device="cuda")
        self.index1 = wp.empty(n, dtype=wp.int64, device="cuda")
        self.index2 = wp.empty(n, dtype=wp.int64, device="cuda")

        self.total_deformed_distance = dist1, dist2 = self.info["deformed_distance"]
        self.target_dist = (dist1 + dist2) * 0.3

        self.mpm_model.struct.particle_type.assign(np.zeros(n, dtype=np.int32))
        self.mpm_model.struct.particle_mass.assign(self.info["model"]["particle_mass"])
        self.mpm_model.struct.particle_vol.assign(self.info["model"]["particle_vol"])
        self.mpm_model.struct.particle_mu_lam_ys.assign(
            self.info["model"]["particle_mu_lam_ys"]
        )

        self.mpm_model.struct.static_ke = 100.0
        self.mpm_model.struct.static_kd = 0.5
        self.mpm_model.struct.static_mu = 0.9
        self.mpm_model.struct.static_ka = 0.0

        self.mpm_model.struct.body_ke = 100.0
        self.mpm_model.struct.body_kd = 0.2
        self.mpm_model.struct.body_mu = 0.5
        self.mpm_model.struct.body_ka = 0.0

        self.mpm_model.adaptive_grid = False
        self.mpm_model.particle_contact = True
        self.mpm_model.grid_contact = False
        self.mpm_model.struct.ground_sticky = True
        self.mpm_model.struct.body_sticky = False

    def _get_coupling_actors(self):
        return [
            l
            for l in self.agent.robot.get_links()
            if l.name in ["panda_hand", "panda_leftfinger", "panda_rightfinger"]
        ]

    def _configure_agent(self):
        self._agent_cfg = PandaPinchConfig()

    def _load_agent(self):
        self.agent = Panda(
            self._scene,
            self.control_freq,
            control_mode=self._control_mode,
            config=self._agent_cfg,
        )
        self.grasp_site: sapien.Link = get_entity_by_name(
            self.agent.robot.get_links(), "panda_hand_tcp"
        )

    def _initialize_agent(self):
        noise = self._episode_rng.uniform([-0.1] * 7 + [0, 0], [0.1] * 7 + [0, 0])
        qpos = np.array([0, 0.01, 0, -1.96, 0.0, 1.98, 0.0, 0.06, 0.06]) + noise

        self.agent.reset(qpos)
        self.agent.robot.set_pose(sapien.Pose([-0.56, 0, 0]))

    def step(self, *args, **kwargs):
        self._chamfer_dist = None
        return super().step(*args, **kwargs)

    def _register_cameras(self):
        p, q = [0.4, 0, 0.3], euler2quat(0, np.pi / 10, -np.pi)
        return CameraConfig("base_camera", p, q, 128, 128, np.pi / 2, 0.001, 10)

    def _register_render_cameras(self):
        p, q = [-0.05, 0.7, 0.3], euler2quat(0, np.pi / 10, -np.pi / 2)
        return CameraConfig("render_camera", p, q, 512, 512, 1, 0.001, 10)

    def compute_dense_reward(self, **kwargs):

        # reaching reward
        gripper_mat = self.grasp_site.get_pose().to_transformation_matrix()
        gripper_bottom_mat = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.02], [0, 0, 0, 1]]
        )
        bottom_mat = gripper_mat @ gripper_bottom_mat
        bottom_pos = sapien.Pose.from_transformation_matrix(bottom_mat).p
        particles_x = self.get_mpm_state()["x"]
        distance = np.min(np.linalg.norm(particles_x - bottom_pos, axis=-1))
        reaching_reward = 1 - np.tanh(10.0 * distance)

        # vertical reward
        gripper_mat = self.grasp_site.pose.to_transformation_matrix()
        z = gripper_mat[:3, 2]
        angle = np.arcsin(np.clip(np.linalg.norm(np.cross(z, [0, 0, -1])), -1, 1))
        reward_orientation = 1 - angle

        # chamfer dist
        if self._chamfer_dist is None:
            n = self.n_particles
            compute_chamfer_distance(
                self.mpm_states[0].struct.particle_q,
                n,
                self.goal_array,
                n,
                self.dist1,
                self.dist2,
                self.index1,
                self.index2,
            )
            wp.synchronize()
            chamfer4_1 = (self.dist1.numpy() ** 2).mean() ** 0.25
            chamfer4_2 = (self.dist2.numpy() ** 2).mean() ** 0.25

            self._chamfer_dist = [chamfer4_1, chamfer4_2]

        return (
            -sum(self._chamfer_dist) * 100.0
            + 0.1 * reaching_reward
            + 0.1 * reward_orientation
        )

    def check_success(self, **kwargs):
        self.compute_dense_reward()
        return sum(self._chamfer_dist) < self.target_dist

    def _get_obs_extra(self):
        target_rgb = self.goal_rgbs
        target_depth = self.goal_depths
        target_points = self.goal_points
        return OrderedDict(
            tcp_pose=vectorize_pose(self.grasp_site.get_pose()),
            target_rgb=target_rgb,
            target_depth=target_depth,
            target_points=target_points,
        )

    def evaluate(self, **kwargs):
        self.compute_dense_reward()
        progress = 1 - sum(self._chamfer_dist) / sum(self.total_deformed_distance)
        # progerss: 0 for initial shape, 1 for target shape, measured with Chamfer
        return {"success": self.check_success(), "progress": progress}

    def get_goal_points(self):
        cam_pos = self.info["goal_cam_pos"]
        cam_rot = self.info["goal_cam_rot"]
        cam_intrinsic = self.info["goal_cam_intrinsic"]

        world_xyzws = []
        total_size = 0
        for depth, rgb, pos, rot in zip(
            self.goal_depths, self.goal_rgbs, cam_pos, cam_rot
        ):
            H, W = depth.shape[:2]
            total_size += H * W
            T = sapien.Pose(pos, rot).to_transformation_matrix()
            xyz = np.stack(
                np.meshgrid(np.arange(0.5, W + 0.5), np.arange(0.5, H + 0.5))
                + [np.ones((H, W))],
                -1,
            )
            cam_xyz = np.linalg.solve(
                cam_intrinsic, (xyz * depth[..., None]).reshape(-1, 3).T
            ).T.reshape(H, W, 3)
            cam_xyz = cam_xyz * [1, -1, -1]  # opencv to opengl
            cam_xyz = (
                cam_xyz @ np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]]).T
            )  # opengl to ros/sapien

            world_xyz = cam_xyz[depth > 0] @ T[:3, :3].T + T[:3, 3]
            w = np.ones(world_xyz.shape[0])
            world_xyzw = np.c_[world_xyz, w]
            world_xyzws.append(world_xyzw)

        points = np.concatenate(world_xyzws, axis=0)
        return np.pad(points, [[0, total_size - len(points)], [0, 0]])
