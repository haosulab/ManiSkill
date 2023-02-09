import os
import numpy as np
import sapien.core as sapien
from mani_skill2.envs.mpm.base_env import MPMBaseEnv
from mani_skill2 import PACKAGE_ASSET_DIR
from mani_skill2.agents.robots.panda import Panda
from mani_skill2.agents.configs.panda.variants import PandaBucketConfig
from mani_skill2.utils.registration import register_env
from mani_skill2.sensors.camera import CameraConfig

from transforms3d.euler import euler2quat
from mani_skill2.utils.sapien_utils import get_entity_by_name
from mani_skill2.utils.sapien_utils import (
    get_entity_by_name,
    vectorize_pose,
)

from collections import OrderedDict

import warp as wp


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
        if z > h1:
            wp.atomic_add(output, 0, 1)
        if z > h2:
            wp.atomic_add(output, 1, 1)
    else:
        # spill
        if z < 0.005:
            wp.atomic_add(output, 2, 1)


@register_env("Fill-v0", max_episode_steps=250)
class FillEnv(MPMBaseEnv):
    def _setup_mpm(self):
        super()._setup_mpm()
        self._success_helper = wp.zeros(3, dtype=int, device=self.mpm_model.device)

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

        x = 0.03
        y = 0.04
        z = 0.03
        cell_x = 0.004
        self.model_builder.add_mpm_grid(
            pos=(-0.2, -0.01, 0.27),
            vel=(0.0, 0.0, 0.0),
            dim_x=int(x // cell_x),
            dim_y=int(y // cell_x),
            dim_z=int(z // cell_x),
            cell_x=cell_x,
            cell_y=cell_x,
            cell_z=cell_x,
            density=3e3,
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
        qpos = np.array([-0.188, 0.234, 0.201, -2.114, -0.088, 1.35, 1.571])
        qpos[-2] += self._episode_rng.normal(0, 0.03, 1)
        self.agent.reset(qpos)
        self.agent.robot.set_pose(sapien.Pose([-0.6, 0, 0]))

    def _register_cameras(self):
        p, q = [-0.4, -0.0, 0.4], euler2quat(0, np.pi / 6, 0)
        return CameraConfig("base_camera", p, q, 128, 128, np.pi / 2, 0.001, 10)

    def _register_render_cameras(self):
        p, q = [-0.5, -0.4, 0.6], euler2quat(0, np.pi / 6, np.pi / 2 - np.pi / 5)
        return CameraConfig("render_camera", p, q, 512, 512, 1, 0.001, 10)

    def _load_actors(self):
        super()._load_actors()
        beaker_file = os.path.join(PACKAGE_ASSET_DIR, "deformable_manipulation", "beaker.glb")

        b = self._scene.create_actor_builder()
        b.add_visual_from_file(beaker_file, scale=[0.04] * 3)
        b.add_collision_from_file(beaker_file, scale=[0.04] * 3, density=300)
        self.target_beaker = b.build_kinematic("target_beaker")

    def _initialize_actors(self):
        super()._initialize_actors()
        random_x = self._episode_rng.rand(1)[0] * 2 - 1
        random_y = self._episode_rng.rand(1)[0] * 2 - 1
        self.beaker_x = -0.16 + random_x * 0.1
        self.beaker_y = random_y * 0.1
        self.target_beaker.set_pose(sapien.Pose([self.beaker_x, self.beaker_y, 0]))

        vs = self.target_beaker.get_visual_bodies()
        assert len(vs) == 1
        v = vs[0]
        vertices = np.concatenate([s.mesh.vertices for s in v.get_render_shapes()], 0)
        self._target_height = vertices[:, 2].max() * v.scale[2]
        self._target_radius = v.scale[0]

    def _get_coupling_actors(
        self,
    ):
        return [
            (l, "visual") for l in self.agent.robot.get_links() if l.name == "bucket"
        ] + [(self.target_beaker, "visual")]

    def _get_obs_extra(self) -> OrderedDict:
        return OrderedDict(
            tcp_pose=vectorize_pose(self.grasp_site.get_pose()),
            target=np.array([self.beaker_x, self.beaker_y]),
        )

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
                0,
                self._target_height,
                self._success_helper,
            ],
            device=self.mpm_model.device,
        )
        above_start, above_end, spill = self._success_helper.numpy()

        if (
            above_start / self.mpm_model.struct.n_particles > 0.9
            and len(np.where((particles_v < 0.05) & (particles_v > -0.05))[0])
            / (self.mpm_model.struct.n_particles * 3)
            > 0.99
        ):
            return dict(success=True)
        return dict(success=False)

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

    def compute_dense_reward(self, reward_info=False, **kwargs):
        if self.evaluate()["success"]:
            if reward_info:
                return {"reward": 2.5}
            return 2.5
        # bucket frame
        gripper_mat = self.grasp_site.get_pose().to_transformation_matrix()
        gripper_bucket_mat = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0.02], [0, 0, 1, 0.08], [0, 0, 0, 1]]
        )
        bucket_mat = gripper_mat @ gripper_bucket_mat
        bucket_pos = sapien.Pose.from_transformation_matrix(bucket_mat).p

        beaker_pos = np.array([self.beaker_x, self.beaker_y])
        dist = np.linalg.norm(bucket_pos[:2] - beaker_pos)
        reaching_reward = 1 - np.tanh(10.0 * dist)

        self._success_helper.zero_()
        wp.launch(
            success_kernel,
            dim=self.mpm_model.struct.n_particles,
            inputs=[
                self.mpm_states[0].struct.particle_q,
                self.target_beaker.pose.p[:2],
                self._target_radius,
                self._target_height,
                0,
                self._target_height,
                self._success_helper,
            ],
            device=self.mpm_model.device,
        )
        above_start, above_end, spill = self._success_helper.numpy()
        inside_reward = above_start / self.n_particles
        spill_reward = -spill / 100

        if reaching_reward > 0.9:
            (
                bucket_base_pos,
                bucket_tlpos,
                bucket_trpos,
                bucket_blpos,
                bucket_brpos,
                gripper_mat,
            ) = self._bucket_keypoints()
            bucket_bottom = (bucket_blpos + bucket_brpos) / 2
            tilt_reward = 1 - np.tanh(100.0 * (bucket_bottom[2] - bucket_base_pos[2]))
        else:
            tilt_reward = 0.4

        if reward_info:
            return {
                "reaching_reward": reaching_reward,
                "inside_reward": inside_reward,
                "spill_reward": spill_reward,
                "tilt_reward": tilt_reward,
            }
        return reaching_reward * 0.1 + inside_reward + spill_reward + tilt_reward * 0.5

    def render(self, mode="human", draw_box=False, draw_target=False):
        if draw_target:
            bbox = self.target_box
            box = self._add_draw_box(bbox)
        img = super().render(mode, draw_box)
        if draw_target:
            self._remove_draw_box(box)
        return img

    def get_state(self) -> np.ndarray:
        state = super().get_state()
        return np.hstack([state, self.beaker_x, self.beaker_y])

    def set_state(self, state):
        self.beaker_x = state[-2]
        self.beaker_y = state[-1]
        super().set_state(state[:-2])


if __name__ == "__main__":

    env = FillEnv()
    env.reset()

    a = env.get_state()
    env.set_state(a)

    for _ in range(100):
        env.step(None)
        env.render(draw_box=True)
