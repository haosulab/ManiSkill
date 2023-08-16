import os
import pickle
import sys
from collections import OrderedDict

import numpy as np
import sapien.core as sapien
import warp as wp
from transforms3d.euler import euler2quat
from transforms3d.quaternions import axangle2quat

from mani_skill2.agents.configs.panda.defaults import PandaDefaultConfig
from mani_skill2.agents.robots.panda import Panda
from mani_skill2.envs.mpm.base_env import MPMBaseEnv, MPMModelBuilder
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import get_entity_by_name, vectorize_pose
from warp_maniskill.mpm.mpm_simulator import Simulator as MPMSimulator


@register_env("Hang-v0", max_episode_steps=350)
class HangEnv(MPMBaseEnv):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        with open(os.path.join(os.path.dirname(__file__), "RopeInit.pkl"), "rb") as f:
            self.rope_init = pickle.load(f)
        super().__init__(*args, **kwargs)

    def _setup_mpm(self):
        self.model_builder = MPMModelBuilder()
        self.model_builder.set_mpm_domain(
            domain_size=[0.5, 0.5, 0.5], grid_length=0.015
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

    def _initialize_mpm(self):
        self.model_builder.clear_particles()

        E = 1e4
        nu = 0.3
        mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))

        # 0 for von-mises, 1 for drucker-prager
        type = 0

        # von-mises
        ys = 1e4

        # drucker-prager
        friction_angle = 0.6
        cohesion = 0.05

        x = 0.4
        y = 0.022
        z = 0.022
        cell_x = 0.004

        self.model_builder.add_mpm_grid(
            pos=(0.1, 0.0, 0.05),
            vel=(0.0, 0.0, 0.0),
            dim_x=int(x // cell_x),
            dim_y=int(y // cell_x),
            dim_z=int(z // cell_x),
            cell_x=cell_x,
            cell_y=cell_x,
            cell_z=cell_x,
            density=300,
            mu_lambda_ys=(mu, lam, ys),
            friction_cohesion=(friction_angle, cohesion, 0.0),
            type=type,
            jitter=True,
            placement_x="center",
            placement_y="center",
            placement_z="start",
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

        self.mpm_model.struct.body_sticky = 1
        self.mpm_model.struct.ground_sticky = 1
        self.mpm_model.particle_contact = True
        self.mpm_model.grid_contact = True

        self.mpm_model.struct.particle_radius = 0.005

        x = self.get_mpm_state()["x"]
        lower = x.min(0)
        upper = x.max(0)
        dim = np.argmax(upper - lower)
        length = upper[dim] - lower[dim]

        # on one side
        mask1 = (lower[dim] + length * 0.05 < x[..., dim]) * (
            x[..., dim] < lower[dim] + length * 0.15
        )

        index1 = np.where(mask1)[0][0]
        mask2 = (lower[dim] + length * 0.25 < x[..., dim]) * (
            x[..., dim] < lower[dim] + length * 0.35
        )
        index2 = np.where(mask2)[0][0]

        # on top
        mask3 = (lower[dim] + length * 0.48 < x[..., dim]) * (
            x[..., dim] < lower[dim] + length * 0.52
        )
        index3 = np.where(mask3)[0][0]

        # on the other side
        mask4 = (lower[dim] + length * 0.65 < x[..., dim]) * (
            x[..., dim] < lower[dim] + length * 0.75
        )
        index4 = np.where(mask4)[0][0]
        mask5 = (lower[dim] + length * 0.85 < x[..., dim]) * (
            x[..., dim] < lower[dim] + length * 0.95
        )
        index5 = np.where(mask5)[0][0]

        self.selected_indices = [index1, index2, index3, index4, index5]

    def _configure_agent(self):
        self._agent_cfg = PandaDefaultConfig()

    def _load_agent(self):
        self.agent = Panda(
            self._scene,
            self._control_freq,
            control_mode=self._control_mode,
            config=self._agent_cfg,
        )
        self.grasp_site: sapien.Link = get_entity_by_name(
            self.agent.robot.get_links(), "panda_hand"
        )

        self.lfinger = get_entity_by_name(
            self.agent.robot.get_links(), "panda_leftfinger"
        )
        self.rfinger = get_entity_by_name(
            self.agent.robot.get_links(), "panda_rightfinger"
        )

    def _load_actors(self):
        super()._load_actors()
        builder = self._scene.create_actor_builder()
        builder.add_box_collision(half_size=[0.3, 0.01, 0.01])
        builder.add_box_visual(half_size=[0.3, 0.01, 0.01], color=[1, 0, 0, 1])
        self.rod = builder.build_kinematic("rod")

    def _register_cameras(self):
        p, q = [0.45, -0.0, 0.5], euler2quat(0, np.pi / 5, np.pi)
        return CameraConfig("base_camera", p, q, 128, 128, np.pi / 2, 0.001, 10)

    def _register_render_cameras(self):
        p, q = [0.2, 1.0, 0.5], euler2quat(0, 0.2, 4.4)
        return CameraConfig("render_camera", p, q, 512, 512, 1, 0.001, 10)

    def initialize_episode(self):
        super().initialize_episode()
        idx = self._episode_rng.randint(len(self.rope_init["mpm_states"]))
        self.agent.set_state(
            self.rope_init["agent_states"][idx], ignore_controller=True
        )
        self.set_mpm_state(self.rope_init["mpm_states"][idx])

    def _initialize_actors(self):
        # default pose
        pose = sapien.Pose([0.1, 0, 0.21], axangle2quat([0, 0, 1], np.pi / 2))

        # random pose
        r = 0.2 + self._episode_rng.rand() * 0.03
        a = np.pi / 4 + self._episode_rng.rand() * np.pi / 2
        x = np.sin(a) * r
        y = np.cos(a) * r
        z = 0.2 + self._episode_rng.rand() * 0.1
        pose = sapien.Pose([x, y, z], axangle2quat([0, 0, 1], -a))
        self.rod.set_pose(pose)

    def _get_coupling_actors(self):
        return self.agent.robot.get_links() + [self.rod]

    def _initialize_agent(self):
        qpos = np.array(
            [0, np.pi / 16, 0, -np.pi * 5 / 6, 0, np.pi - 0.2, np.pi / 4, 0, 0]
        )
        self.agent.reset(qpos)
        self.agent.robot.set_pose(sapien.Pose([-0.46, 0, 0]))

    def _get_obs_extra(self) -> OrderedDict:
        return OrderedDict(
            tcp_pose=vectorize_pose(self.grasp_site.get_pose()),
            target=np.hstack([self.rod.get_pose().p, self.rod.get_pose().q]),
        )

    def evaluate(self, **kwargs):
        particles_x = self.get_mpm_state()["x"]
        particles_v = self.get_mpm_state()["v"]

        lf_pos = self.lfinger.get_pose().p
        rf_pos = self.rfinger.get_pose().p
        finger_dist = np.linalg.norm(lf_pos - rf_pos)

        pose = self.rod.pose
        center = pose.p
        normal = pose.to_transformation_matrix()[:3, :3] @ np.array([0, 1, 0])

        x = particles_x[self.selected_indices]
        dirs = x - center

        signs = np.sign(dirs @ normal)

        side = signs[0] == signs[1] and signs[3] == signs[4] and signs[0] != signs[3]
        top = (
            np.max(particles_x[:, 2]) > center[2]
            and np.max(particles_x[:, 2]) < center[2] + 0.05
        )
        down = dirs[0, 2] < 0 and dirs[4, 2] < 0
        bottom = np.min(particles_x[:, 2]) > 0.03

        high_ind = np.where(particles_x[:, 2] > center[2] - 0.03)
        particles_v = particles_v[high_ind]
        return dict(
            success=(
                side
                and top
                and down
                and bottom
                and len(np.where((particles_v < 0.05) & (particles_v > -0.05))[0])
                / (len(particles_v) * 3 + 0.001)
                > 0.99
                and finger_dist > 0.07
            )
        )

    def compute_dense_reward(self, reward_info=False, **kwargs):
        gripper_width = (
            self.agent.robot.get_qlimits()[-1, 1] * 2
        )  # NOTE: hard-coded with panda
        if self.evaluate()["success"]:
            reward = 6
            reaching_reward = 1
            center_reward = 1
            side_reward = 1
            top_reward = 1
            release_reward = 1
        else:
            # reaching reward
            gripper_pos = self.grasp_site.get_pose().p
            particles_x = self.get_mpm_state()["x"]
            distance = np.min(np.linalg.norm(particles_x - gripper_pos, axis=-1))
            reaching_reward = 1 - np.tanh(10.0 * distance)

            # rope center reward
            center_reward = 0.0
            pose = self.rod.pose
            rod_center = pose.p
            rope_center = particles_x[self.selected_indices[2]]

            distance = np.linalg.norm(rod_center[:2] - rope_center[:2])
            center_reward += 0.5 * (1 - np.tanh(10.0 * distance))

            if rod_center[2] >= rope_center[2]:
                distance = rod_center[2] - rope_center[2]
                center_reward += 0.5 * (1 - np.tanh(10.0 * distance))
            else:
                center_reward += 0.5

            # rope pos reward
            top_reward = 0
            bottom_reward = 0
            release_reward = 0
            if rod_center[2] >= rope_center[2]:
                side_reward = 0
            else:
                bottom_reward = 1 - np.tanh(
                    10.0 * max(0, 0.04 - np.min(particles_x[:, 2]))
                )
                rod_normal = pose.to_transformation_matrix()[:3, :3] @ np.array(
                    [0, 1, 0]
                )
                x = particles_x[self.selected_indices]
                dirs = x - rod_center
                signs = np.sign(dirs @ rod_normal)

                side_reward = 0.5 * (
                    int(signs[0] != signs[3])
                    + int(
                        signs[0] == signs[1]
                        and signs[3] == signs[4]
                        and signs[0] != signs[3]
                    )
                )

                if (
                    signs[0] == signs[1]
                    and signs[3] == signs[4]
                    and signs[0] != signs[3]
                ):
                    top_reward = 0.25 * (
                        int(dirs[0, 2] < 0)
                        + int(dirs[1, 2] < 0)
                        + int(dirs[3, 2] < 0)
                        + int(dirs[4, 2] < 0)
                    )
                    reaching_reward = 1
                    if top_reward > 0.9:
                        release_reward = (
                            np.sum(self.agent.robot.get_qpos()[-2:]) / gripper_width
                        )
            reward = (
                reaching_reward
                + center_reward
                + side_reward
                + top_reward
                + release_reward
                + bottom_reward * 0.2
            )

        if reward_info:
            return {
                "reward": reward,
                "reaching_reward": reaching_reward,
                "center_reward": center_reward,
                "side_reward": side_reward,
                "top_reward": top_reward,
                "release_reward": release_reward,
                "bottom_reward": bottom_reward,
                "bottom_pos": np.min(particles_x[:, 2]),
            }
        return reward

    def compute_normalized_dense_reward(self, **kwargs):
        return self.compute_dense_reward(**kwargs) / 6.0

    def get_state(self) -> np.ndarray:
        state = super().get_state()
        return np.hstack([state, self.rod.get_pose().p, self.rod.get_pose().q])

    def set_state(self, state):
        pose = sapien.Pose(state[-7:-4], state[-4:])
        self.rod.set_pose(pose)
        super().set_state(state[:-7])


if __name__ == "__main__":
    env = HangEnv(reward_mode="dense", obs_mode="pointcloud")
    agent_states = []
    soft_states = []

    a = env.get_state()
    env.set_state(a)

    for X in np.linspace(-0.081, -0.021, 10):
        print(X)
        env.reset()
        env.agent.set_control_mode("pd_ee_delta_pos")
        env.step(np.array([X, 0, 1, 1]))
        for _ in range(10):
            env.step(None)
            env.render()

        env.step(np.array([0, 0, 0, -1]))
        for _ in range(20):
            env.step(None)
            env.render()

        for i in range(50):
            env.step(np.array([0, 0, -0.2 * i / 100, -1]))
            env.render()
        agent_states.append(env.agent.get_state())
        print(env.agent.get_state().keys())
        soft_states.append(env.get_mpm_state())

    # with open("RopeInit.pkl", "wb") as f:
    #     pickle.dump({"mpm_states": soft_states, "agent_states": agent_states}, f)
