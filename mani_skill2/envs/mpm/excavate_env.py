from collections import OrderedDict

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat

from mani_skill2.agents.configs.panda.variants import PandaBucketConfig
from mani_skill2.agents.robots.panda import Panda
from mani_skill2.envs.mpm import perlin
from mani_skill2.envs.mpm.base_env import MPMBaseEnv
from mani_skill2.envs.mpm.utils import actor2meshes
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import get_entity_by_name, vectorize_pose


@register_env("Excavate-v0", max_episode_steps=250)
class ExcavateEnv(MPMBaseEnv):
    def __init__(
        self,
        *args,
        sim_freq=500,
        mpm_freq=2000,
        **kwargs,
    ):
        super().__init__(
            *args,
            sim_freq=sim_freq,
            mpm_freq=mpm_freq,
            **kwargs,
        )

    def _initialize_mpm(self):
        self.model_builder.clear_particles()

        E = 1e4
        nu = 0.3
        mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))

        # 0 for von-mises, 1 for drucker-prager
        type = 1

        # von-mises
        ys = 1e4

        # drucker-prager
        friction_angle = 0.6
        cohesion = 0.05

        height_map = 0.06 + perlin.added_perlin(
            [0.03, 0.02, 0.02],
            [1, 2, 4],
            phases=[(0, 0), (0, 0), (0, 0)],
            shape=(30, 30),
            random_state=self._episode_rng,
        )

        count = self.model_builder.add_mpm_from_height_map(
            pos=(0.0, 0.0, -0.0),
            vel=(0.0, 0.0, 0.0),
            dx=0.005,
            height_map=height_map,
            density=3.0e3,
            mu_lambda_ys=(mu, lam, ys),
            friction_cohesion=(friction_angle, cohesion, 0.0),
            type=type,
            jitter=True,
            color=(1, 1, 0.5),
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

        self.mpm_model.adaptive_grid = True

        self.mpm_model.grid_contact = True
        self.mpm_model.particle_contact = True
        self.mpm_model.struct.body_sticky = 1
        self.mpm_model.struct.ground_sticky = 1

        self.mpm_model.struct.particle_radius = 0.0025

    def _configure_agent(self):
        self._agent_cfg = PandaBucketConfig()

    def _load_agent(self):
        self.agent = Panda(
            self._scene,
            self._control_freq,
            control_mode=self._control_mode,
            config=self._agent_cfg,
        )

        self.grasp_site: sapien.Link = get_entity_by_name(
            self.agent.robot.get_links(), "bucket"
        )

    def _initialize_agent(self):
        qpos = np.array([-0.174, 0.457, 0.203, -1.864, -0.093, 2.025, 1.588])
        qpos[:-1] += self._episode_rng.normal(0, 0.02, len(qpos) - 1)
        qpos[-1] += self._episode_rng.normal(0, 0.2, 1)
        self.agent.reset(qpos)
        self.agent.robot.set_pose(sapien.Pose([-0.56, 0, 0]))
        self.home_ee = np.array([0.0, 0.0, self.target_height])

    def _initialize_actors(self):
        super()._initialize_actors()
        self.target_height = 0.2
        self.target_num = self._episode_rng.choice(range(250, 1150), 1)[0]
        self.mpm_model.struct.n_particles = len(self.model_builder.mpm_particle_q)

        bucket_mesh = actor2meshes(self.grasp_site)[0]
        vertices = bucket_mesh.vertices
        ones = np.ones(len(vertices))
        self.vertices_mat = np.column_stack((vertices, ones.T))

    def _load_actors(self):
        super()._load_actors()
        b = self._scene.create_actor_builder()
        b.add_box_collision(half_size=[0.12, 0.02, 0.03])
        b.add_box_visual(half_size=[0.12, 0.02, 0.03])
        w0 = b.build_kinematic("wall")
        w1 = b.build_kinematic("wall")
        w2 = b.build_kinematic("wall")
        w3 = b.build_kinematic("wall")

        w0.set_pose(sapien.Pose([0, -0.1, 0.03]))
        w1.set_pose(sapien.Pose([0, 0.1, 0.03]))
        w2.set_pose(sapien.Pose([-0.1, 0, 0.03], [0.7071068, 0, 0, 0.7071068]))
        w3.set_pose(sapien.Pose([0.1, 0, 0.03], [0.7071068, 0, 0, 0.7071068]))
        self.walls = [w0, w1, w2, w3]

    def _get_coupling_actors(
        self,
    ):
        return [
            (l, "visual") for l in self.agent.robot.get_links() if l.name == "bucket"
        ] + self.walls

    def _register_cameras(self):
        p, q = [-0.2, -0, 0.4], euler2quat(0, np.pi / 6, 0)
        return CameraConfig("base_camera", p, q, 128, 128, np.pi / 2, 0.001, 10)

    def _register_render_cameras(self):
        p, q = [-0.35, -0, 0.4], euler2quat(0, np.pi / 6, 0)
        return CameraConfig("render_camera", p, q, 512, 512, 1, 0.001, 10)

    def _get_obs_extra(self) -> OrderedDict:
        return OrderedDict(
            tcp_pose=vectorize_pose(self.grasp_site.get_pose()),
            target=np.array([self.target_num]),
        )

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        target_num = options.pop("target_num", None)
        ret = super().reset(seed=seed, options=options)
        if target_num is not None:
            self.target_num = int(target_num)
        return ret

    def evaluate(self, **kwargs):
        particles_x = self.get_mpm_state()["x"]
        particles_v = self.get_mpm_state()["v"]
        lift_num = len(np.where(particles_x[:, 2] > self.target_height)[0])
        spill_num = self.mpm_model.struct.n_particles - len(
            np.where(
                (particles_x[:, 0] > -0.12)
                & (particles_x[:, 0] < 0.12)
                & (particles_x[:, 1] > -0.12)
                & (particles_x[:, 1] < 0.12)
            )[0]
        )
        return dict(
            success=(
                lift_num > self.target_num - 100
                and lift_num < self.target_num + 150
                and spill_num < 20
                and len(np.where((particles_v < 0.05) & (particles_v > -0.05))[0])
                / (self.mpm_model.struct.n_particles * 3)
                > 0.99
            )
        )

    def _in_bbox_ids(self, particles_x, bbox):
        return np.where(
            (particles_x[:, 0] >= np.min(bbox[:, 0]))
            & (particles_x[:, 0] <= np.max(bbox[:, 0]))
            & (particles_x[:, 1] >= np.min(bbox[:, 1]))
            & (particles_x[:, 1] <= np.max(bbox[:, 1]))
            & (particles_x[:, 2] >= np.min(bbox[:, 2]))
            & (particles_x[:, 2] <= np.max(bbox[:, 2]))
        )[0]

    def _in_bucket_ids(self, particles_x, bbox, top_signs, bot_signs):
        return np.where(
            (particles_x[:, 0] >= np.min(bbox[:, 0]))
            & (particles_x[:, 0] <= np.max(bbox[:, 0]))
            & (particles_x[:, 1] >= np.min(bbox[:, 1]))
            & (particles_x[:, 1] <= np.max(bbox[:, 1]))
            & (particles_x[:, 2] >= np.min(bbox[:, 2]))
            & (particles_x[:, 2] <= np.max(bbox[:, 2]))
            & (top_signs > 0)
            & (bot_signs > 0)
        )[0]

    def _bucket_keypoints(self):
        gripper_mat = self.grasp_site.get_pose().to_transformation_matrix()
        bucket_base_mat = np.array(
            [[1, 0, 0, 0], [0, 1, 0, -0.01], [0, 0, 1, 0.045], [0, 0, 0, 1]]
        )
        bucket_tlmat = np.array(
            [[1, 0, 0, -0.03], [0, 1, 0, -0.01], [0, 0, 1, 0.01], [0, 0, 0, 1]]
        )
        bucket_trmat = np.array(
            [[1, 0, 0, 0.03], [0, 1, 0, -0.01], [0, 0, 1, 0.01], [0, 0, 0, 1]]
        )
        bucket_blmat = np.array(
            [[1, 0, 0, -0.03], [0, 1, 0, 0.02], [0, 0, 1, 0.08], [0, 0, 0, 1]]
        )
        bucket_brmat = np.array(
            [[1, 0, 0, 0.03], [0, 1, 0, 0.02], [0, 0, 1, 0.08], [0, 0, 0, 1]]
        )
        bucket_base_pos = sapien.Pose.from_transformation_matrix(
            gripper_mat @ bucket_base_mat
        ).p
        bucket_tlpos = sapien.Pose.from_transformation_matrix(
            gripper_mat @ bucket_tlmat
        ).p
        bucket_trpos = sapien.Pose.from_transformation_matrix(
            gripper_mat @ bucket_trmat
        ).p
        bucket_blpos = sapien.Pose.from_transformation_matrix(
            gripper_mat @ bucket_blmat
        ).p
        bucket_brpos = sapien.Pose.from_transformation_matrix(
            gripper_mat @ bucket_brmat
        ).p
        return (
            bucket_base_pos,
            bucket_tlpos,
            bucket_trpos,
            bucket_blpos,
            bucket_brpos,
            gripper_mat,
        )

    def _get_bbox(self, points):
        return np.array(
            [
                [np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2])],
                [np.max(points[:, 0]), np.max(points[:, 1]), np.max(points[:, 2])],
            ]
        )

    def bucket_top_normal(self):
        (
            bucket_base_pos,
            bucket_tlpos,
            bucket_trpos,
            bucket_blpos,
            bucket_brpos,
            gripper_mat,
        ) = self._bucket_keypoints()
        bucket_top_normal = np.cross(
            bucket_base_pos - bucket_trpos, bucket_trpos - bucket_tlpos
        )
        bucket_top_normal /= np.linalg.norm(bucket_top_normal)
        return bucket_top_normal

    def particles_inside_bucket(self):
        # bucket boundary
        (
            bucket_base_pos,
            bucket_tlpos,
            bucket_trpos,
            bucket_blpos,
            bucket_brpos,
            gripper_mat,
        ) = self._bucket_keypoints()

        bucket_top_normal = np.cross(
            bucket_base_pos - bucket_trpos, bucket_trpos - bucket_tlpos
        )
        bucket_top_normal /= np.linalg.norm(bucket_top_normal)
        top_d = -np.sum(bucket_top_normal * bucket_base_pos)
        top_vec = np.array(list(bucket_top_normal) + [top_d])
        bucket_bot_normal = np.cross(
            bucket_base_pos - bucket_blpos, bucket_blpos - bucket_brpos
        )
        bucket_bot_normal /= np.linalg.norm(bucket_bot_normal)
        bot_d = -np.sum(bucket_bot_normal * bucket_base_pos)
        bot_vec = np.array(list(bucket_bot_normal) + [bot_d])

        vertices = (gripper_mat @ self.vertices_mat.T).T[:, :3]
        bbox = self._get_bbox(vertices)

        # pick particles
        particles_x = self.get_mpm_state()["x"]
        ones = np.ones(len(particles_x))
        particles = np.column_stack((particles_x, ones.T))
        top_signs = particles @ top_vec.T
        bot_signs = particles @ bot_vec.T
        lifted_particles = particles_x[
            self._in_bucket_ids(particles_x, bbox, top_signs, bot_signs)
        ]
        return lifted_particles

    def compute_dense_reward(self, reward_info=False, **kwargs):
        if self.evaluate()["success"]:
            if reward_info:
                return {"reward": 6.0}
            return 6.0
        particles_x = self.get_mpm_state()["x"]

        stage = 0

        # spill reward
        spill_num = self.n_particles - len(
            np.where(
                (particles_x[:, 0] > -0.12)
                & (particles_x[:, 0] < 0.12)
                & (particles_x[:, 1] > -0.12)
                & (particles_x[:, 1] < 0.12)
            )[0]
        )
        spill_reward = -spill_num / 100

        (
            bucket_base_pos,
            bucket_tlpos,
            bucket_trpos,
            bucket_blpos,
            bucket_brpos,
            gripper_mat,
        ) = self._bucket_keypoints()

        lifted_particles = self.particles_inside_bucket()
        lift_num = len(lifted_particles)
        lift_reward = (
            min(lift_num / self.target_num, 1)
            - max(0, lift_num - self.target_num - 500) * 0.001
        )

        gripper_pos = self.grasp_site.get_pose().p
        height_dist = (
            max(self.target_height + 0.05 - np.mean(lifted_particles[:, 2]), 0)
            if len(lifted_particles) > 0
            else 1
        )
        # reaching reward & height reward & flat reward
        if height_dist > 0.1 and lift_num > self.target_num + 300:
            reaching_reward = 1
            height_reward = 1 - np.tanh(3 * height_dist)
            flat_dist = 0.5 * (
                max(bucket_base_pos[2] + 0.01 - bucket_blpos[2], 0)
                + max(bucket_blpos[2] - bucket_brpos[2], 0)
            )
            flat_reward = 1 - np.tanh(50 * flat_dist)
            stage = 1
        elif height_dist <= 0.1:
            lift_reward = (
                1
                + min(lift_num / self.target_num, 1)
                - max(0, lift_num - self.target_num - 100) * 0.001
            )
            reaching_reward = 1
            height_reward = 1 - np.tanh(3 * height_dist)
            flat_dist = 0.5 * (
                max(bucket_base_pos[2] - 0.01 - bucket_blpos[2], 0)
                + max(bucket_blpos[2] - bucket_brpos[2], 0)
            )
            flat_reward = 1 - np.tanh(50 * flat_dist)
            stage = 2
        else:
            if (
                gripper_pos[0] > -0.1
                and gripper_pos[0] < 0.1
                and gripper_pos[1] > -0.1
                and gripper_pos[1] < 0.1
            ):
                dist = gripper_pos[2] + max(0.04 - gripper_pos[0], 0)
                reaching_reward = 1 - np.tanh(10 * dist)
            else:
                reaching_reward = 0
            height_reward = 0
            flat_reward = 0

        reward = (
            reaching_reward * 0.5
            + lift_reward
            + height_reward
            + spill_reward
            + flat_reward
        )
        if reward_info:
            return {
                "reward": reward,
                "reaching_reward": reaching_reward,
                "lift_reward": lift_reward,
                "lift_num": lift_num,
                "target_num": self.target_num,
                "flat_reward": flat_reward,
                "height_reward": height_reward,
                "spill_reward": spill_reward,
                "stage": stage,
                "height_dist": height_dist,
            }
        return reward

    def compute_normalized_dense_reward(self, **kwargs):
        return self.compute_dense_reward(**kwargs) / 6.0

    def render(self, draw_box=False, draw_target=False):
        if draw_target:
            bbox = self.target_box
            box = self._add_draw_box(bbox)

        img = super().render(draw_box)
        if draw_target:
            self._remove_draw_box(box)
        return img

    def get_state(self) -> np.ndarray:
        state = super().get_state()
        return np.hstack([state, self.target_num])

    def set_state(self, state):
        self.target_num = state[-1]
        super().set_state(state[:-1])


if __name__ == "__main__":
    env = ExcavateEnv(reward_mode="dense")
    env.reset()
    env.agent.set_control_mode("pd_ee_delta_pose")

    a = env.get_state()
    env.set_state(a)

    for i in range(100):
        env.step(None)
        env.render()
