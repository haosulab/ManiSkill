import warp as wp
import numpy as np
from mpm.mpm_model import MPMModelBuilder, mpm_collide, Mesh, DenseVolume
from mpm.mpm_integrator import (
    compute_grid_bound,
    set_grid_bound,
    g2p,
    p2g,
    grid_op,
    grid_op_with_contact,
    create_soft_contacts,
    eval_soft_contacts,
    MPMModel,
    MPMState,
    zero_everything,
)
import os


class Simulator:
    def __init__(self, device="cuda"):
        self.device = device

    def simulate(
        self, model: MPMModel, state_in: MPMState, state_out: MPMState, dt: float
    ):
        wp.launch(
            zero_everything,
            dim=int(
                max(
                    model.body_count,
                    model.struct.n_particles,
                    model.struct.grid_dim_x
                    * model.struct.grid_dim_y
                    * model.struct.grid_dim_z,
                )
            ),
            inputs=[
                state_in.struct,
                state_in.ext_body_f,
                state_in.int_body_f,
                state_in.mpm_contact_count,
                model.struct.grid_dim_x,
                model.struct.grid_dim_y,
                model.struct.grid_dim_z,
                model.struct.n_particles,
                model.body_count,
            ],
            device=self.device,
        )

        if model.adaptive_grid:
            wp.launch(
                compute_grid_bound,
                dim=int(model.struct.n_particles),
                inputs=[model.struct, state_in.struct],
                device=self.device,
            )
        else:
            wp.launch(
                set_grid_bound,
                dim=1,
                inputs=[model.struct, state_in.struct],
                device=self.device,
            )

        if model.shape_count > 0:
            wp.launch(
                create_soft_contacts,
                dim=int(model.struct.n_particles * model.shape_count),
                inputs=[
                    model.struct.n_particles,
                    state_in.struct.particle_q,
                    state_in.body_q,
                    model.shape_transform,
                    model.shape_body,
                    model.shape_geo_type,
                    model.shape_geo_id,
                    model.shape_geo_scale,
                    model.mpm_contact_margin,
                    state_in.mpm_contact_count,
                    state_in.mpm_contact_particle,
                    state_in.mpm_contact_body,
                    state_in.mpm_contact_body_pos,
                    state_in.mpm_contact_body_vel,
                    state_in.mpm_contact_normal,
                    model.mpm_contact_max,
                ],
                device=self.device,
            )

            wp.launch(
                eval_soft_contacts,
                dim=int(model.mpm_contact_max),
                inputs=[
                    model.struct,
                    state_in.struct,
                    state_in.body_q,
                    state_in.body_qd,
                    model.body_com,
                    state_in.mpm_contact_count,
                    state_in.mpm_contact_particle,
                    state_in.mpm_contact_body,
                    state_in.mpm_contact_body_pos,
                    state_in.mpm_contact_body_vel,
                    state_in.mpm_contact_normal,
                    model.struct.particle_radius,
                    state_in.ext_body_f,
                ],
                device=self.device,
            )

        wp.launch(
            p2g,
            dim=int(model.struct.n_particles),
            inputs=[
                model.struct,
                state_in.struct,
                state_out.struct,
                model.gravity,
                dt,
            ],
            device=self.device,
        )

        if model.body_count > 0 and model.grid_contact:
            wp.launch(
                grid_op_with_contact,
                dim=int(
                    model.struct.grid_dim_x
                    * model.struct.grid_dim_y
                    * model.struct.grid_dim_z
                ),
                inputs=[
                    model.struct,
                    state_in.struct,
                    dt,
                    state_in.body_q,
                    state_in.body_qd,
                    model.body_com,
                    model.shape_transform,
                    model.shape_body,
                    model.shape_geo_type,
                    model.shape_geo_id,
                    model.shape_geo_scale,
                    model.shape_count,
                    state_in.ext_body_f,
                ],
                device=self.device,
            )
        else:
            wp.launch(
                grid_op,
                dim=int(
                    model.struct.grid_dim_x
                    * model.struct.grid_dim_y
                    * model.struct.grid_dim_z
                ),
                inputs=[
                    model.struct,
                    state_in.struct,
                    dt,
                ],
                device=self.device,
            )

        wp.launch(
            g2p,
            dim=int(model.struct.n_particles),
            inputs=[model.struct, state_in.struct, state_out.struct, dt],
            device=self.device,
        )

class App:
    def __init__(self, stage=None):
        self.render_scale = 100
        wp.init()

        if stage:
            self.renderer = wp.render.UsdRenderer(stage)
        else:
            self.renderer = None

        self.device = "cuda"
        self.sim_dt = 5e-4

        builder = MPMModelBuilder()
        builder.set_mpm_domain([1.0, 1.0, 1.0], 0.005)
        E = 1e5
        nu = 0.3
        ys = 10000.0
        mu, lam = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
        cohesion = 0.1
        friction_angle = np.pi / 3

        count = builder.add_mpm_grid(
            pos=(0.0, 0.2, 0.0),
            vel=(0.0, 0.0, 0.0),
            dim_x=30,
            dim_y=15,
            dim_z=30,
            cell_x=0.005,
            cell_y=0.005,
            cell_z=0.005,
            density=3e3,
            mu_lambda_ys=(mu, lam, ys),
            friction_cohesion=(friction_angle, cohesion, 0.0),
            type=0,
            jitter=True,
            placement_x="center",
            placement_y="corner",
            placement_z="center",
            color=(125 / 255, 87 / 255, 0),
        )

        import trimesh

        def trimesh2sdf(meshes, margin, dx, bbox=None):
            if meshes is None:
                return None
            mesh = trimesh.util.concatenate(meshes)

            if bbox is None:
                bbox = mesh.bounds.copy()

            sdfs = []
            normals = []
            for mesh in meshes:
                center = (bbox[0] + bbox[1]) / 2
                res = np.ceil((bbox[1] - bbox[0] + margin * 2) / dx).astype(int)
                lower = center - res * dx / 2.0

                points = np.zeros((res[0], res[1], res[2], 3))
                x = np.arange(0.5, res[0]) * dx + lower[0]
                y = np.arange(0.5, res[1]) * dx + lower[1]
                z = np.arange(0.5, res[2]) * dx + lower[2]

                points[..., 0] += x[:, None, None]
                points[..., 1] += y[None, :, None]
                points[..., 2] += z[None, None, :]

                points = points.reshape((-1, 3))

                query = trimesh.proximity.ProximityQuery(mesh)
                sdf = query.signed_distance(points) * -1.0

                surface_points, _, tri_id = query.on_surface(points)
                face_normal = mesh.face_normals[tri_id]
                normal = (points - surface_points) * np.sign(sdf)[..., None]
                length = np.linalg.norm(normal, axis=-1)
                mask = length < 1e6
                normal[mask] = face_normal[mask]
                normal = normal / (
                    np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8
                )
                sdf = sdf.reshape(res)
                normal = normal.reshape((res[0], res[1], res[2], 3))

                sdfs.append(sdf)
                normals.append(normal)

            if len(sdfs) == 1:
                sdf = sdfs[0]
                normal = normals[0]
            else:
                sdfs = np.stack(sdfs)
                normals = np.stack(normals)
                index = np.expand_dims(sdfs.argmin(0), 0)
                sdf = np.take_along_axis(sdfs, index, 0)[0]
                normal = np.take_along_axis(normals, np.expand_dims(index, -1), 0)[0]

            return {
                "sdf": sdf,
                "normal": normal,
                "position": lower,
                "scale": np.ones(3) * dx,
                "dim": res,
            }

        box = trimesh.creation.box((0.05, 0.1, 0.05))
        sdf = trimesh2sdf([box], 0.02, 0.01)

        volume = DenseVolume(
            np.concatenate([sdf["normal"], sdf["sdf"][..., None]], -1),
            position=sdf["position"],
            scale=sdf["scale"],
            mesh=Mesh(box.vertices, box.faces.flatten()),
        )

        body = builder.add_body(wp.transform((0.0, 0.5, 0.0)))
        builder.add_shape_dense_volume(body, volume=volume, density=2e3)

        # builder.add_shape_box(
        #     -1, pos=(0, 0 + 0.5, 0), rot=(0.1305262, 0, 0, 0.9914449), hy=0.05
        # )

        self.model = builder.finalize(self.device)

        self.model.struct.particle_radius = 0.005
        self.model.mpm_contact_distance = 0.005
        self.model.mpm_contact_margin = 0.01

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        builder.clear_particles()
        builder.add_mpm_grid(
            pos=(0.0, 0.2, 0.0),
            vel=(0.0, 0.0, 0.0),
            dim_x=15,
            dim_y=15,
            dim_z=15,
            cell_x=0.005,
            cell_y=0.005,
            cell_z=0.005,
            density=3e3,
            mu_lambda_ys=(mu, lam, ys),
            friction_cohesion=(friction_angle, cohesion, 0.0),
            type=0,
            jitter=True,
            placement_x="center",
            placement_y="corner",
            placement_z="center",
            color=(125 / 255, 87 / 255, 0),
        )
        builder.init_model_state(self.model, self.state_0)

        self.sim = Simulator(self.device)
        self.sim_time = 0.0

        if self.renderer:
            self.renderer.render_ground(
                self.render_scale / 2, color=(0.005, 0.01, 0.01)
            )

    def update(self):
        pass
        self.sim.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0

        prev_frame = int(self.sim_time * 60)

        self.sim_time += self.sim_dt

        this_frame = int(self.sim_time * 60)

        if self.render and prev_frame != this_frame:
            self.render()

        wp.synchronize()

    def render(self):
        if self.renderer:
            self.renderer.begin_frame(self.sim_time)

            body_q = (
                self.state_0.body_q.numpy() if self.state_0.body_q is not None else None
            )
            shape_body = self.model.shape_body.numpy()
            shape_transform = self.model.shape_transform.numpy()
            shape_geo_scale = self.model.shape_geo_scale.numpy()

            for shape_id, (body_id, T_bs, scale, src) in enumerate(
                zip(
                    shape_body,
                    shape_transform,
                    shape_geo_scale,
                    self.model.shape_geo_src,
                )
            ):
                T_wb = (
                    np.array([0, 0, 0, 0, 0, 0, 1]) if body_id < 0 else body_q[body_id]
                )
                T_ws = wp.transform_multiply(
                    wp.transform_expand(T_wb), wp.transform_expand(T_bs)
                )

                if isinstance(src, DenseVolume):
                    self.renderer.render_mesh(
                        f"shape_{shape_id}",
                        src.mesh.vertices,
                        src.mesh.indices,
                        np.array(T_ws.p) * self.render_scale,
                        np.array(T_ws.q),
                        scale * self.render_scale,
                    )
                else:
                    self.renderer.render_box(
                        f"shape_{shape_id}",
                        np.array(T_ws.p) * self.render_scale,
                        np.array(T_ws.q),
                        scale * self.render_scale,
                    )

            self.renderer.render_points(
                "points",
                points=self.state_0.struct.particle_q.numpy()[
                    : self.model.struct.n_particles
                ]
                * self.render_scale,
                radius=float(self.model.struct.particle_radius) * self.render_scale,
            )
            self.renderer.end_frame()


if __name__ == "__main__":
    wp.init()
    stage_path = os.path.join(os.path.dirname(__file__), "out.usd")
    app = App(stage_path)
    from tqdm import trange

    for _ in trange(4000):
        app.update()
    app.renderer.save()
