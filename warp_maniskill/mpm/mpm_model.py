from warp.sim.model import Model, ModelBuilder, Vec3, Mesh, DenseVolume
from warp.sim.collide import create_soft_contacts

import warp as wp
import numpy as np


@wp.struct
class MPMModelStruct:
    n_particles: int
    dx: float
    inv_dx: float
    grid_dim_x: int
    grid_dim_y: int
    grid_dim_z: int
    particle_radius: float

    ground_normal: wp.vec3

    static_ke: float
    static_kd: float
    static_mu: float
    static_ka: float
    ground_sticky: int

    body_ke: float
    body_kd: float
    body_mu: float
    body_ka: float
    body_sticky: int

    particle_vol: wp.array(dtype=float)  # init volume
    particle_mass: wp.array(dtype=float)
    particle_mu_lam_ys: wp.array(dtype=wp.vec3)
    particle_friction_cohesion: wp.array(dtype=wp.vec3)
    particle_type: wp.array(dtype=int)


@wp.struct
class MPMStateStruct:
    grid_lower: wp.array(dtype=int)
    grid_upper: wp.array(dtype=int)

    particle_q: wp.array(dtype=wp.vec3)
    particle_qd: wp.array(dtype=wp.vec3)
    particle_F: wp.array(dtype=wp.mat33)
    particle_C: wp.array(dtype=wp.mat33)

    particle_vol: wp.array(dtype=float)  # current volume
    particle_volume_correction: wp.array(dtype=float)
    particle_f: wp.array(dtype=wp.vec3)

    grid_m: wp.array(dtype=float, ndim=3)
    grid_mv: wp.array(dtype=wp.vec3, ndim=3)
    grid_v: wp.array(dtype=wp.vec3, ndim=3)
    error: wp.array(dtype=int)


class MPMState:
    def __init__(self, model):
        self.model = model
        self.struct = MPMStateStruct()

    def clear_forces(self):
        if self.model.body_count:
            self.ext_body_f.zero_()
            self.int_body_f.zero_()


class MPMModel(Model):
    def __init__(self, device):
        super().__init__(device)
        self.mpm_particle_q = None
        self.mpm_particle_qd = None
        self.struct = MPMModelStruct()
        self.struct.ground_normal = (0.0, 1.0, 0.0)
        self.adaptive_grid = False

        self.struct.static_ke = 100.0
        self.struct.static_kd = 0.0
        self.struct.static_ka = 0.0
        self.struct.static_mu = 0.5

        self.struct.body_ke = 100.0
        self.struct.body_kd = 0.0
        self.struct.body_ka = 0.0
        self.struct.body_mu = 0.5

        self.struct.ground_sticky = False
        self.struct.body_sticky = False
        self.particle_contact = True
        self.grid_contact = True

    def state(self, requires_grad=False) -> MPMState:
        s = MPMState(self)
        s.struct.grid_lower = wp.zeros(
            3, dtype=int, device=self.device, requires_grad=False
        )
        s.struct.grid_upper = wp.zeros(
            3, dtype=int, device=self.device, requires_grad=False
        )

        # --------------------------------
        s.body_q = None
        s.body_qd = None
        s.int_body_f = None
        s.ext_body_f = None

        # no particles
        # if self.particle_count:
        #     s.particle_q = wp.clone(self.particle_q)
        #     s.particle_qd = wp.clone(self.particle_qd)
        #     s.particle_q.requires_grad = requires_grad
        #     s.particle_qd.requires_grad = requires_grad

        # articulations
        if self.body_count:
            s.body_q = wp.clone(self.body_q)
            s.body_qd = wp.clone(self.body_qd)
            s.int_body_f = wp.zeros_like(self.body_qd)
            s.ext_body_f = wp.zeros_like(self.body_qd)

            s.body_q.requires_grad = requires_grad
            s.body_qd.requires_grad = requires_grad
            s.int_body_f.requires_grad = requires_grad
            s.ext_body_f.requires_grad = requires_grad

        if self.mpm_particle_q is not None:
            s.struct.particle_q = wp.clone(self.mpm_particle_q)
            s.struct.particle_q.requires_grad = requires_grad
            s.struct.particle_qd = wp.clone(self.mpm_particle_qd)
            s.struct.particle_qd.requires_grad = requires_grad

            s.struct.particle_vol = wp.clone(self.struct.particle_vol)
            s.struct.particle_vol.requires_grad = requires_grad

            eye = np.array([np.eye(3, dtype=np.float32)] * self.max_n_particles)
            s.struct.particle_F = wp.array(
                eye, dtype=wp.mat33, device=self.device, requires_grad=requires_grad
            )

            s.struct.particle_volume_correction = wp.zeros(
                self.max_n_particles,
                dtype=float,
                device=self.device,
                requires_grad=requires_grad,
            )

            # ∇v
            s.struct.particle_C = wp.zeros(
                self.max_n_particles,
                dtype=wp.mat33,
                device=self.device,
                requires_grad=requires_grad,
            )

            s.struct.error = wp.zeros(
                1, dtype=int, device=self.device, requires_grad=False
            )

            s.struct.grid_v = wp.zeros(
                (
                    self.struct.grid_dim_x,
                    self.struct.grid_dim_y,
                    self.struct.grid_dim_z,
                ),
                dtype=wp.vec3,
                device=self.device,
                requires_grad=requires_grad,
            )

            s.struct.grid_mv = wp.zeros(
                (
                    self.struct.grid_dim_x,
                    self.struct.grid_dim_y,
                    self.struct.grid_dim_z,
                ),
                dtype=wp.vec3,
                device=self.device,
                requires_grad=requires_grad,
            )

            s.struct.grid_m = wp.zeros(
                (
                    self.struct.grid_dim_x,
                    self.struct.grid_dim_y,
                    self.struct.grid_dim_z,
                ),
                dtype=float,
                device=self.device,
                requires_grad=requires_grad,
            )
            s.struct.particle_f = wp.zeros(
                self.max_n_particles, dtype=wp.vec3, device=self.device
            )

            s.mpm_contact_count = wp.zeros(1, dtype=wp.int32, device=self.device)
            s.mpm_contact_particle = wp.zeros(
                self.mpm_contact_max, dtype=int, device=self.device
            )
            s.mpm_contact_body = wp.zeros(
                self.mpm_contact_max, dtype=int, device=self.device
            )
            s.mpm_contact_body_pos = wp.zeros(
                self.mpm_contact_max, dtype=wp.vec3, device=self.device
            )
            s.mpm_contact_body_vel = wp.zeros(
                self.mpm_contact_max, dtype=wp.vec3, device=self.device
            )
            s.mpm_contact_normal = wp.zeros(
                self.mpm_contact_max, dtype=wp.vec3, device=self.device
            )

        return s


class MPMModelBuilder(ModelBuilder):
    def __init__(self):
        super().__init__()
        self.mpm_particle_mass = []
        self.mpm_particle_q = []
        self.mpm_particle_qd = []
        self.mpm_particle_volume = []
        self.mpm_particle_mu_lam_ys = []
        self.mpm_particle_friction_cohesion = []
        self.mpm_particle_type = []
        self.mpm_particle_colors = []

        self.mpm_grid_length = None
        self.mpm_domain_dims = None

    def set_mpm_domain(self, domain_size, grid_length=0.01):
        domain_dims = (np.array(domain_size) / grid_length).astype(int)
        self.mpm_domain_dims = domain_dims
        self.mpm_grid_length = grid_length

    def add_mpm_cylinder(
        self,
        pos: Vec3,
        vel: Vec3,
        radius: float,
        height: float,
        dx: float,
        density: float,
        mu_lambda_ys: Vec3,
        friction_cohesion: Vec3,
        type: int,
        jitter=False,
        color=(1, 1, 1),
        random_state=np.random.RandomState(),
    ):
        # rejection sample the unit circle
        radius = radius // dx * dx
        line = np.arange(-radius, radius + 0.5 * dx, dx)
        points = np.stack(np.meshgrid(line, line), -1).reshape((-1, 2))
        mask = points[:, 0] ** 2 + points[:, 1] ** 2 < radius**2
        xy = points[mask]

        height = height // dx * dx
        zs = np.arange(0, height + 0.5 * dx, dx)

        points = np.array([[x, y, z] for x, y in xy for z in zs])

        if jitter:
            points += (random_state.random((len(xy) * len(zs), 3)) - 0.5) * dx

        volume = np.pi * radius * radius * height / len(points)
        mass = volume * density

        for p in points:
            self.add_mpm_particle(
                (pos[0] + p[0], pos[1] + p[1], pos[2] + p[2]),
                vel,
                mass=mass,
                volume=volume,
                type=type,
                material=mu_lambda_ys,
                material2=friction_cohesion,
                color=color,
            )

        return len(self.mpm_particle_q)

    def add_mpm_from_height_map(
        self,
        pos: Vec3,
        vel: Vec3,
        dx: float,
        height_map: np.ndarray,
        density: float,
        mu_lambda_ys: Vec3,
        friction_cohesion: Vec3,
        type: int,
        jitter=False,
        color=(1, 1, 1),
        random_state=np.random.RandomState(),
    ):
        h, w = height_map.shape
        hh = (h - 1) * dx / 2
        hw = (w - 1) * dx / 2
        points = np.stack(
            np.meshgrid(
                np.arange(-hw, hw + 0.5 * dx, dx),
                np.arange(-hh, hh + 0.5 * dx, dx),
            ),
            -1,
        )
        volume = dx**3
        mass = volume * density
        for (x, y), h in zip(points.reshape((-1, 2)), height_map.reshape(-1)):
            for z in np.arange(0, h, dx):
                jx = 0
                jy = 0
                jz = 0
                if jitter:
                    jx = random_state.uniform(-0.5, 0.5) * dx
                    jy = random_state.uniform(-0.5, 0.5) * dx
                    jz = random_state.uniform(-0.5, 0.5) * dx

                self.add_mpm_particle(
                    (pos[0] + x + jx, pos[1] + y + jy, pos[2] + z + jz),
                    vel,
                    mass=mass,
                    volume=volume,
                    type=type,
                    material=mu_lambda_ys,
                    material2=friction_cohesion,
                    color=color,
                )

        return len(self.mpm_particle_q)

    def add_mpm_grid(
        self,
        pos: Vec3,
        vel: Vec3,
        dim_x: int,
        dim_y: int,
        dim_z: int,
        cell_x: float,
        cell_y: float,
        cell_z: float,
        density: float,
        mu_lambda_ys: Vec3,
        friction_cohesion: Vec3,
        type: int,
        jitter=False,
        placement_x="center",
        placement_y="start",
        placement_z="center",
        fix_px=False,
        fix_nx=False,
        fix_py=False,
        fix_ny=False,
        fix_pz=False,
        fix_nz=False,
        color=(1, 1, 1),
        random_state=np.random.RandomState(),
    ):
        pos = np.array(pos)
        if placement_x == "center":
            pos[0] = pos[0] - cell_x * dim_x / 2
        if placement_y == "center":
            pos[1] = pos[1] - cell_y * dim_y / 2
        if placement_z == "center":
            pos[2] = pos[2] - cell_z * dim_z / 2

        volume = cell_x * cell_y * cell_z
        mass = volume * density

        if jitter:
            jitter = random_state.random((dim_x + 1, dim_y + 1, dim_z + 1, 3)) - 0.5
        else:
            jitter = np.zeros((dim_x + 1, dim_y + 1, dim_z + 1, 3))

        for z in range(dim_z + 1):
            for y in range(dim_y + 1):
                for x in range(dim_x + 1):
                    j = jitter[x, y, z]
                    v = np.array(
                        ((x + j[0]) * cell_x, (y + j[1]) * cell_y, (z + j[2]) * cell_z)
                    )
                    m = mass
                    p = v + pos

                    fix = (
                        (x == 0 and fix_nx)
                        or (x == dim_x and fix_px)
                        or (y == 0 and fix_ny)
                        or (y == dim_y and fix_py)
                        or (z == 0 and fix_nz)
                        or (z == dim_z and fix_pz)
                    )

                    self.add_mpm_particle(
                        p,
                        vel,
                        mass=m,
                        volume=volume,
                        type=type,
                        material=mu_lambda_ys,
                        material2=friction_cohesion,
                        color=color,
                    )

        return len(self.mpm_particle_q)

    def add_mpm_particle(
        self,
        pos: Vec3,
        vel: Vec3,
        mass: float,
        volume: float,
        type: int,
        material: Vec3 = (0.0, 0.0, 0.0),
        material2: Vec3 = (0.5, 0.0, 0.0),
        color=(1.0, 1.0, 1.0),
    ) -> int:

        self.mpm_particle_q.append(pos)
        self.mpm_particle_qd.append(vel)
        self.mpm_particle_mass.append(mass)
        self.mpm_particle_volume.append(volume)
        self.mpm_particle_mu_lam_ys.append(material)
        self.mpm_particle_friction_cohesion.append(material2)
        self.mpm_particle_type.append(type)
        self.mpm_particle_colors.append(color)

        return len(self.mpm_particle_q) - 1

    def clear_particles(self):
        self.mpm_particle_q = []
        self.mpm_particle_qd = []
        self.mpm_particle_mass = []
        self.mpm_particle_volume = []
        self.mpm_particle_mu_lam_ys = []
        self.mpm_particle_friction_cohesion = []
        self.mpm_particle_type = []
        self.mpm_particle_colors = []

    def reserve_mpm_particles(self, count):
        self.mpm_particle_q += [(0.0, 0.0, 0.0)] * count
        self.mpm_particle_qd += [(0.0, 0.0, 0.0)] * count
        self.mpm_particle_mass += [1.0] * count
        self.mpm_particle_volume += [1.0] * count
        self.mpm_particle_mu_lam_ys += [(0.0, 0.0, 0.0)] * count
        self.mpm_particle_friction_cohesion += [(0.0, 0.0, 0.0)] * count
        self.mpm_particle_type += [0] * count
        self.mpm_particle_colors += [(1.0, 1.0, 1.0)] * count

        return len(self.mpm_particle_q) - 1

    def finalize(self, device: str) -> MPMModel:
        assert (
            self.mpm_domain_dims is not None
        ), "set_mpm_domain must be called before calling finalize"

        m0: MPMModel = super().finalize(device)
        m = MPMModel(device)

        for k, v in vars(m0).items():
            setattr(m, k, v)

        m.max_n_particles = len(self.mpm_particle_q)
        m.struct.n_particles = len(self.mpm_particle_q)
        m.struct.particle_radius = 0.1

        mpm_particle_q = np.array(self.mpm_particle_q, dtype=np.float32)
        mpm_particle_qd = np.array(self.mpm_particle_qd, dtype=np.float32)
        mpm_particle_mass = np.array(self.mpm_particle_mass, dtype=np.float32)
        mpm_particle_volume = np.array(self.mpm_particle_volume, dtype=np.float32)
        mpm_particle_mu_lam_ys = np.array(self.mpm_particle_mu_lam_ys, dtype=np.float32)
        mpm_particle_friction_cohesion = np.array(
            self.mpm_particle_friction_cohesion, dtype=np.float32
        )
        mpm_particle_type = np.array(self.mpm_particle_type, dtype=np.int32)
        mpm_particle_colors = np.array(self.mpm_particle_colors, dtype=np.float32)

        m.mpm_particle_colors = mpm_particle_colors

        m.mpm_particle_q = wp.array(mpm_particle_q, dtype=wp.vec3, device=device)
        m.mpm_particle_qd = wp.array(mpm_particle_qd, dtype=wp.vec3, device=device)

        m.struct.particle_mass = wp.array(
            mpm_particle_mass, dtype=wp.float32, device=device
        )
        m.struct.particle_vol = wp.array(
            mpm_particle_volume, dtype=wp.float32, device=device
        )
        m.struct.particle_mu_lam_ys = wp.array(
            mpm_particle_mu_lam_ys, dtype=wp.vec3, device=device
        )
        m.struct.particle_friction_cohesion = wp.array(
            mpm_particle_friction_cohesion, dtype=wp.vec3, device=device
        )

        m.struct.particle_type = wp.array(
            mpm_particle_type, dtype=wp.int32, device=device
        )
        m.struct.grid_dim_x = self.mpm_domain_dims[0]
        m.struct.grid_dim_y = self.mpm_domain_dims[1]
        m.struct.grid_dim_z = self.mpm_domain_dims[2]
        m.struct.dx = self.mpm_grid_length
        m.struct.inv_dx = 1.0 / self.mpm_grid_length

        m.mpm_contact_max = 64 * 1024
        m.mpm_contact_distance = 0.01
        m.mpm_contact_margin = 0.1

        return m

    def init_model_state(self, model: MPMModel, states):
        assert len(self.mpm_particle_q) <= model.max_n_particles
        model.struct.n_particles = len(self.mpm_particle_q)
        model.struct.particle_vol.assign(
            np.array(self.mpm_particle_volume, dtype=np.float32)
        )
        model.struct.particle_mass.assign(
            np.array(self.mpm_particle_mass, dtype=np.float32)
        )
        model.struct.particle_mu_lam_ys.assign(
            np.array(self.mpm_particle_mu_lam_ys, dtype=np.float32)
        )
        model.struct.particle_friction_cohesion.assign(
            np.array(self.mpm_particle_friction_cohesion, dtype=np.float32)
        )
        model.struct.particle_type.assign(np.array(self.mpm_particle_type, dtype=int))
        model.mpm_particle_colors = np.array(self.mpm_particle_colors, dtype=np.float32)

        for state in states:
            state.struct.particle_q.assign(
                np.array(self.mpm_particle_q, dtype=np.float32)
            )
            state.struct.particle_qd.assign(
                np.array(self.mpm_particle_qd, dtype=np.float32)
            )

            eye = np.array([np.eye(3, dtype=np.float32)] * model.max_n_particles)
            state.struct.particle_F.assign(eye)
            state.struct.particle_volume_correction.zero_()
            state.struct.particle_C.zero_()
            state.struct.particle_vol.assign(
                np.array(self.mpm_particle_volume, dtype=np.float32)
            )

            state.struct.grid_lower.zero_()
            state.struct.grid_upper.zero_()
            state.struct.particle_f.zero_()
            state.struct.grid_m.zero_()
            state.struct.grid_mv.zero_()
            state.struct.grid_v.zero_()
            state.struct.error.zero_()


def mpm_collide(model: MPMModel, state: MPMState):
    state.mpm_contact_count.zero_()
    wp.launch(
        kernel=create_soft_contacts,
        dim=model.mpm_particle_count * model.shape_count,
        inputs=[
            model.mpm_particle_count,
            state.mpm_particle_q,
            state.body_q,
            model.shape_transform,
            model.shape_body,
            model.shape_geo_type,
            model.shape_geo_id,
            model.shape_geo_scale,
            model.mpm_contact_margin,
            state.mpm_contact_count,
            state.mpm_contact_particle,
            state.mpm_contact_body,
            state.mpm_contact_body_pos,
            state.mpm_contact_body_vel,
            state.mpm_contact_normal,
            model.mpm_contact_max,
        ],
        outputs=[],
        device=model.device,
    )
