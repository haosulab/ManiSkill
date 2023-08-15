import os
from collections import OrderedDict

import numpy as np
import sapien.core as sapien
import warp as wp
from transforms3d.euler import euler2quat

from mani_skill2 import PACKAGE_ASSET_DIR
from mani_skill2.agents.configs.panda.variants import PandaPourConfig
from mani_skill2.agents.robots.panda import Panda
from mani_skill2.envs.mpm.base_env import MPMBaseEnv, MPMModelBuilder, MPMSimulator
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.geometry import (
    get_local_aabc_for_actor,
    get_local_axis_aligned_bbox_for_link,
)
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import get_entity_by_name, vectorize_pose


@wp.kernel
def success_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    center: wp.vec2,
    radius: float,
    height: float,
    h1: float,
    h2: float,
    output: wp.array(dtype=int),
):
    tid = wp.tid()
    x = particle_q[tid]
    a = x[0] - center[0]
    b = x[1] - center[1]
    z = x[2]
    if a * a + b * b < radius * radius and z < height:
        wp.atomic_add(output, 3, 1)
        if z > h1:
            wp.atomic_add(output, 0, 1)
        if z > h2:
            wp.atomic_add(output, 1, 1)
    else:
        # spill
        if z < 0.001:
            wp.atomic_add(output, 2, 1)


def create_ring():
    segs = 16
    angles = np.linspace(0, 2 * np.pi, segs, endpoint=False)
    xs = np.cos(angles)
    ys = np.sin(angles)
    vs = np.zeros((segs, 3))
    vs[:, 0] = xs
    vs[:, 1] = ys

    vs2 = vs.copy()
    vs2[:, 2] = 1

    vertices = np.concatenate([vs, vs2], 0)
    indices = []
    for i in range(segs):
        a = i
        b = (i + 1) % segs
        c = b + segs
        d = a + segs
        indices.append(a)
        indices.append(b)
        indices.append(c)
        indices.append(a)
        indices.append(c)
        indices.append(d)

        indices.append(a)
        indices.append(c)
        indices.append(b)
        indices.append(a)
        indices.append(d)
        indices.append(c)

    return vertices, np.array(indices)


@register_env("Pour-v0", max_episode_steps=350)
class PourEnv(MPMBaseEnv):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self.robot_uid = "panda"
        self._ring = None
        super().__init__(*args, **kwargs)

    def _load_actors(self):
        super()._load_actors()
        bottle_file = os.path.join(
            PACKAGE_ASSET_DIR, "deformable_manipulation", "bottle.glb"
        )
        beaker_file = os.path.join(
            PACKAGE_ASSET_DIR, "deformable_manipulation", "beaker.glb"
        )

        b = self._scene.create_actor_builder()
        b.add_visual_from_file(bottle_file, scale=[0.025] * 3)
        b.add_collision_from_file(bottle_file, scale=[0.025] * 3, density=300)
        self.source_container = b.build("bottle")
        self.source_aabb = get_local_axis_aligned_bbox_for_link(self.source_container)

        target_radius = 0.04
        b = self._scene.create_actor_builder()
        b.add_visual_from_file(beaker_file, scale=[target_radius] * 3)
        b.add_collision_from_file(beaker_file, scale=[target_radius] * 3, density=300)
        self.target_beaker = b.build_kinematic("target_beaker")
        self.target_aabb = get_local_axis_aligned_bbox_for_link(self.target_beaker)
        self.target_aabc = get_local_aabc_for_actor(self.target_beaker)

    def _get_coupling_actors(
        self,
    ):
        return [
            (self.source_container, "visual"),
            (self.target_beaker, "visual"),
        ]

    def _configure_agent(self):
        self._agent_cfg = PandaPourConfig()

    def _load_agent(self):
        self.agent = Panda(
            self._scene,
            self._control_freq,
            control_mode=self._control_mode,
            config=self._agent_cfg,
        )
        self.grasp_site: sapien.Link = get_entity_by_name(
            self.agent.robot.get_links(), "panda_hand_tcp"
        )
        self.lfinger = get_entity_by_name(
            self.agent.robot.get_links(), "panda_leftfinger"
        )
        self.rfinger = get_entity_by_name(
            self.agent.robot.get_links(), "panda_rightfinger"
        )

    def _setup_mpm(self):
        self.model_builder = MPMModelBuilder()
        self.model_builder.set_mpm_domain(
            domain_size=[0.8, 0.8, 0.8], grid_length=0.005
        )
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
        self._success_helper = wp.zeros(4, dtype=int, device=self.mpm_model.device)

    def _initialize_mpm(self):
        self.model_builder.clear_particles()

        E = 3e5
        nu = 0.1
        mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
        ys = 0.0
        type = 2

        viscosity = self._episode_rng.uniform(0.0, 3.0)
        density = self._episode_rng.uniform(0.8e3, 2.0e3)

        count = self.model_builder.add_mpm_cylinder(
            pos=(*self._source_pos.p[:2], 0.01),
            vel=(0.0, 0.0, 0.0),
            radius=0.024,
            height=self._episode_rng.uniform(0.07, 0.09),
            dx=0.0025,
            density=density,
            mu_lambda_ys=(mu, lam, ys),
            friction_cohesion=(0.0, 0.0, viscosity),
            type=type,
            jitter=False,
            random_state=self._episode_rng,
            color=[0.0, 0.5, 0.8],
        )

        for i in range(len(self.model_builder.mpm_particle_volume)):
            self.model_builder.mpm_particle_volume[i] *= 1.2

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

        self.mpm_model.struct.body_sticky = 0
        self.mpm_model.struct.ground_sticky = 1
        self.mpm_model.particle_contact = True
        self.mpm_model.grid_contact = False
        self.mpm_model.struct.particle_radius = 0.0025

    def _initialize_actors(self):
        super()._initialize_actors()

        self._determine_target_pos()

        self.target_beaker.set_pose(self._target_pos)
        self.source_container.set_pose(self._source_pos)

        self.source_container.set_velocity([0, 0, 0])
        self.source_container.set_angular_velocity([0, 0, 0])

        vs = self.target_beaker.get_visual_bodies()
        assert len(vs) == 1
        v = vs[0]
        vertices = np.concatenate([s.mesh.vertices for s in v.get_render_shapes()], 0)
        self._target_height = vertices[:, 2].max() * v.scale[2]
        self._target_radius = v.scale[0]

    def _initialize_agent(self):
        self.agent.reset(self._init_qpos)
        self.agent.robot.set_pose(sapien.Pose([-0.55, 0, 0]))

    def _register_cameras(self):
        p, q = [0.4, 0, 0.3], euler2quat(0, np.pi / 10, -np.pi)
        return CameraConfig("base_camera", p, q, 128, 128, np.pi / 2, 0.001, 10)

    def _register_render_cameras(self):
        p, q = [-0.05, 0.7, 0.3], euler2quat(0, np.pi / 10, -np.pi / 2)
        return CameraConfig("render_camera", p, q, 512, 512, 1, 0.001, 10)

    def initialize_episode(self):
        super().initialize_episode()

        self.h1 = self._episode_rng.uniform(0.01, 0.02)
        self.h2 = self.h1 + 0.004

        if self._ring is not None:
            self._scene.remove_actor(self._ring)

        vertices, indices = create_ring()
        b = self._scene.create_actor_builder()
        mesh = self._renderer.create_mesh(
            vertices.reshape((-1, 3)), indices.reshape((-1, 3))
        )
        mat = self._renderer.create_material()
        mat.set_base_color([1, 0, 0, 1])
        b.add_visual_from_mesh(
            mesh,
            scale=[
                self._target_radius * 1.02,
                self._target_radius * 1.02,
                self.h2 - self.h1,
            ],
            material=mat,
        )
        ring = b.build_kinematic("ring")
        ring.set_pose(sapien.Pose([*self.target_beaker.pose.p[:2], self.h1]))
        self._ring = ring

    def _clear(self):
        if self._ring is not None:
            self._scene.remove_actor(self._ring)
            self._ring = None
        super()._clear()

    def _setup_viewer(self):
        super()._setup_viewer()
        self._viewer.set_camera_xyz(-0.05, 0.3, 0.3)
        self._viewer.set_camera_rpy(0.0, -0.7, 1.57)

    def _determine_target_pos(self):
        pmodel = self.agent.robot.create_pinocchio_model()
        hand = next(
            l for l in self.agent.robot.get_links() if l.name == "panda_hand_tcp"
        )
        while True:
            r = self._episode_rng.uniform(0.2, 0.25)
            t = self._episode_rng.uniform(0, np.pi)
            self._target_pos = sapien.Pose([r * np.cos(t), r * np.sin(t), 0.0])

            r = self._episode_rng.uniform(0.05, 0.1)
            t = self._episode_rng.uniform(np.pi, np.pi * 2)
            self._source_pos = sapien.Pose([r * np.cos(t), r * np.sin(t), 0.0])

            from transforms3d.quaternions import axangle2quat, qmult

            q = qmult(
                axangle2quat(
                    [0, 0, 1], self._episode_rng.uniform(-np.pi / 8, np.pi / 8)
                ),
                [0.5, -0.5, -0.5, -0.5],
            )

            result, success, error = pmodel.compute_inverse_kinematics(
                hand.get_index(),
                sapien.Pose(
                    [
                        self._source_pos.p[0] + 0.55,
                        self._source_pos.p[1],
                        self._episode_rng.uniform(0.04, 0.06),
                    ],
                    q,
                ),
                [-0.555, 0.646, 0.181, -1.892, 1.171, 1.423, -1.75, 0.04, 0.04],
                active_qmask=[1] * 7 + [0] * 2,
            )
            if not success:
                continue

            result[-2:] = 0.04
            self._init_qpos = result
            return

    def _get_obs_extra(self) -> OrderedDict:
        return OrderedDict(
            tcp_pose=vectorize_pose(self.grasp_site.get_pose()),
            target=np.array([self.h1]),
        )

    def in_beaker_num(self):
        self._success_helper.zero_()
        wp.launch(
            success_kernel,
            dim=self.mpm_model.struct.n_particles,
            inputs=[
                self.mpm_states[0].struct.particle_q,
                self.target_beaker.pose.p[:2],
                self._target_radius,
                self._target_height,
                self.h1,
                self.h2,
                self._success_helper,
            ],
            device=self.mpm_model.device,
        )
        above_start, above_end, spill, in_beaker = self._success_helper.numpy()
        return above_start, above_end

    def evaluate(self, **kwargs):
        particles_v = self.get_mpm_state()["v"]
        self._success_helper.zero_()
        wp.launch(
            success_kernel,
            dim=self.mpm_model.struct.n_particles,
            inputs=[
                self.mpm_states[0].struct.particle_q,
                self.target_beaker.pose.p[:2],
                self._target_radius,
                self._target_height,
                self.h1,
                self.h2,
                self._success_helper,
            ],
            device=self.mpm_model.device,
        )
        above_start, above_end, spill, in_beaker = self._success_helper.numpy()

        upright = (
            self.source_container.pose.to_transformation_matrix()[:3, 2]
            @ np.array([0, 0, 1])
        ) >= 0.866  # 30 degrees

        if (
            above_start > 100
            and above_end < 10
            and spill < 100
            and upright
            and np.max(self.agent.robot.get_qvel()) < 0.05
            and np.min(self.agent.robot.get_qvel()) > -0.05
        ):
            return dict(success=True)
        return dict(success=False)

    def compute_dense_reward(self, reward_info=False, **kwargs):
        if self.evaluate()["success"]:
            if reward_info:
                return {"reward": 15}
            return 15
        self._success_helper.zero_()
        wp.launch(
            success_kernel,
            dim=self.mpm_model.struct.n_particles,
            inputs=[
                self.mpm_states[0].struct.particle_q,
                self.target_beaker.pose.p[:2],
                self._target_radius,
                self._target_height,
                self.h1,
                self.h2,
                self._success_helper,
            ],
            device=self.mpm_model.device,
        )
        above_start, above_end, spill, in_beaker = self._success_helper.numpy()

        source_mat = self.source_container.pose.to_transformation_matrix()
        target_mat = self.target_beaker.pose.to_transformation_matrix()

        a, b = self.source_aabb
        t = np.array([0.5, 0.5, 0.33])
        grasp_site_target = a * (1 - t) + b * t
        grasp_site_target = source_mat[:3, :3] @ grasp_site_target + source_mat[:3, 3]
        grasp_site_dist = np.linalg.norm(self.grasp_site.pose.p - grasp_site_target)
        reward_grasp_site = -grasp_site_dist

        if grasp_site_dist < 0.05:
            reward_grasp_site = 0
            check_grasp = self.agent.check_grasp(self.source_container)
            reward_grasp = float(check_grasp)
        else:
            check_grasp = False
            reward_grasp = 0

        top_center = np.zeros(3)
        top_center[:2] = (a[:2] + b[:2]) * 0.5
        top_center[2] = b[2]
        top_center = source_mat[:3, :3] @ top_center + source_mat[:3, 3]

        bottom_center = np.zeros(3)
        bottom_center[:2] = (a[:2] + b[:2]) * 0.5
        bottom_center[2] = a[2]
        bottom_center = source_mat[:3, :3] @ bottom_center + source_mat[:3, 3]

        tx, ty, tr, tzmin, tzmax = self.target_aabc
        target_top_center = (
            target_mat[:3, :3] @ np.array([tx, ty, tzmax]) + target_mat[:3, 3]
        )

        dist_lf = np.linalg.norm(self.lfinger.pose.p[:2] - target_top_center[:2])
        dist_rf = np.linalg.norm(self.rfinger.pose.p[:2] - target_top_center[:2])
        reward_finger = 10 * (dist_rf - dist_lf)

        hdist = np.linalg.norm(target_top_center[:2] - self.grasp_site.pose.p[:2])
        vdist = target_top_center[2] - self.grasp_site.pose.p[2]

        reward_in_beaker = 0
        if above_start < 100 or (above_start > 100 and above_start - above_end < 100):
            done = False
            reward_in_beaker = in_beaker
        else:
            done = True
            reward_in_beaker = in_beaker - max(0, above_start - 500) * 2

        reward_spill = -spill

        stage = 0
        z = source_mat[:3, 2]
        edist = 0
        bot_hdist = 0
        if not check_grasp:
            # not grasping the bottle
            reward_orientation = 0
            reward_dist = 0
            stage = 0
        elif done:
            # finish pouring
            angle = np.arcsin(np.clip(np.linalg.norm(np.cross(z, [0, 0, 1])), -1, 1))
            reward_orientation = 3.5 - angle
            reward_dist = 5.5 - np.tanh(10.0 * (bottom_center[2] - top_center[2]))
            reward_finger = 1
            stage = 3
        elif vdist > -0.06:
            # looking for the right range
            angle = np.arcsin(np.clip(np.linalg.norm(np.cross(z, [0, 0, 1])), -1, 1))
            reward_orientation = 1 - angle
            reward_dist = 1 - np.tanh(10.0 * (max(0, vdist + 0.06)))
            stage = 1
        else:
            # within the right range to pour

            top_dist = np.linalg.norm(top_center[:2] - target_top_center[:2])
            bottom_dist = np.linalg.norm(bottom_center[:2] - target_top_center[:2])

            reward_dist = 2 - np.tanh(
                10.0 * np.linalg.norm(top_center - target_top_center)
            )
            if dist_rf - dist_lf > 0:
                reward_finger = max(
                    10 * (dist_rf - dist_lf), 10 * (bottom_dist - top_dist)
                )

            reward_orientation = 2 - np.tanh(
                10.0 * max(0, top_center[2] - bottom_center[2])
            )
            stage = 2

        if reward_info:
            return {
                "reward": reward_grasp_site
                + reward_grasp
                + reward_in_beaker * 0.001
                + reward_spill * 0.01
                + reward_orientation
                + reward_dist
                + reward_finger,
                "reward_grasp_site": reward_grasp_site,
                "reward_grasp": reward_grasp,
                "reward_in_beaker": reward_in_beaker,
                "reward_spill": reward_spill,
                "reward_orientation": reward_orientation,
                "reward_dist": reward_dist,
                "stage": stage,
                "hdist": hdist,
                "vdist": vdist,
                "tilt": top_center[2] - bottom_center[2],
                "edist": edist,
                "above_start": above_start,
                "in_beaker": in_beaker,
                "above_end": above_end,
                "tr": tr,
                "reward_finger": reward_finger,
            }
        return (
            reward_grasp_site
            + reward_grasp
            + reward_in_beaker * 0.001
            + reward_spill * 0.01
            + reward_orientation
            + reward_dist
            + reward_finger
        )

    def compute_normalized_dense_reward(self, **kwargs):
        return self.compute_dense_reward(**kwargs) / 15.0

    def get_mpm_state(self):
        n = self.mpm_model.struct.n_particles

        return OrderedDict(
            x=self.copy_array_to_numpy(self.mpm_states[0].struct.particle_q, n),
            v=self.copy_array_to_numpy(self.mpm_states[0].struct.particle_qd, n),
            F=self.copy_array_to_numpy(self.mpm_states[0].struct.particle_F, n),
            C=self.copy_array_to_numpy(self.mpm_states[0].struct.particle_C, n),
            vol=self.copy_array_to_numpy(self.mpm_states[0].struct.particle_vol, n),
        )

    def set_mpm_state(self, state):
        self.mpm_states[0].struct.particle_q.assign(state["x"])
        self.mpm_states[0].struct.particle_qd.assign(state["v"])
        self.mpm_states[0].struct.particle_F.assign(state["F"])
        self.mpm_states[0].struct.particle_C.assign(state["C"])
        self.mpm_states[0].struct.particle_vol.assign(state["vol"])

    def get_state(self) -> np.ndarray:
        state = super().get_state()
        return np.hstack(
            [
                state,
                self.h1,
                self.target_beaker.pose.p,
                self.target_beaker.pose.q,
                self.source_container.pose.p,
                self.source_container.pose.q,
            ]
        )

    def set_state(self, state):
        source_pos = sapien.Pose(state[-7:-4], state[-4:])
        target_pos = sapien.Pose(state[-14:-11], state[-11:-7])
        self.source_container.set_pose(source_pos)
        self.target_beaker.set_pose(target_pos)
        self.h1 = state[-15]

        state = state[:-15]
        sim_state = OrderedDict()
        mpm_state = OrderedDict()
        n = self.mpm_model.struct.n_particles

        sim_state["sapien"] = state[: -n * 25]
        mpm_state["x"] = state[-n * 25 : -n * 22].reshape((n, 3))
        mpm_state["v"] = state[-n * 22 : -n * 19].reshape((n, 3))
        mpm_state["F"] = state[-n * 19 : -n * 10].reshape((n, 3, 3))
        mpm_state["C"] = state[-n * 10 : -n].reshape((n, 3, 3))
        mpm_state["vol"] = state[-n:].reshape((n,))
        sim_state["mpm"] = mpm_state

        return self.set_sim_state(sim_state)


if __name__ == "__main__":
    env = PourEnv(mpm_freq=2000)
    env.reset()
    env.agent.set_control_mode("pd_ee_delta_pose")

    a = env.get_state()
    env.set_state(a)

    for i in range(100):
        env.step(None)
        env.render()
