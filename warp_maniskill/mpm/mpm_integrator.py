import warp as wp
from mpm.mpm_model import MPMModel, MPMState, MPMModelStruct, MPMStateStruct

from warp.sim.collide import (
    sphere_sdf,
    sphere_sdf_grad,
    box_sdf,
    box_sdf_grad,
    capsule_sdf,
    capsule_sdf_grad,
)


@wp.func
def PK1_to_cauchy(P: wp.mat33, F: wp.mat33, J: float):
    return P * wp.transpose(F) * (1.0 / J)


@wp.func
def compute_von_mises(
    F: wp.mat33, U: wp.mat33, s: wp.vec3, V: wp.mat33, ys: float, mu: float
):
    s = wp.vec3(wp.max(0.01, s[0]), wp.max(0.01, s[1]), wp.max(0.01, s[2]))
    e = wp.vec3(wp.log(s[0]), wp.log(s[1]), wp.log(s[2]))

    logdetF = e[0] + e[1] + e[2]  # log particle volume before projection

    tre = e[0] + e[1] + e[2]
    eh = e + wp.vec3(tre / -3.0)
    eh_norm = wp.length(eh) + 1e-6
    delta_gamma = eh_norm - ys / (2.0 * mu)
    if delta_gamma > 0.0:
        e = e - (delta_gamma / eh_norm) * eh
        logdetF_new = e[0] + e[1] + e[2]
        s_new = wp.vec3(wp.exp(e[0]), wp.exp(e[1]), wp.exp(e[2]))
        F_new = U * wp.diag(s_new) * wp.transpose(V)
    else:
        s_new = s
        F_new = F
        logdetF_new = logdetF

        vc = logdetF_new - logdetF

    return F_new, vc


@wp.func
def compute_drucker_prager(
    F: wp.mat33,
    U: wp.mat33,
    s: wp.vec3,
    V: wp.mat33,
    mu: float,
    lam: float,
    friction_angle: float,
    cohesion: float,
    vc: float,
):
    s = wp.vec3(wp.max(0.01, s[0]), wp.max(0.01, s[1]), wp.max(0.01, s[2]))
    e = wp.vec3(wp.log(s[0]), wp.log(s[1]), wp.log(s[2]))

    sa = wp.sin(friction_angle)
    alpha = 1.632993161855452 * sa / (3.0 - sa)

    logdetF = e[0] + e[1] + e[2]  # log particle volume before projection

    # volume change negative ⇒ forget about expansion ⇒ add expanded volume back
    e = e - wp.vec3(wp.min(vc, 0.0) / 3.0)

    tre = e[0] + e[1] + e[2]
    eh = e + wp.vec3(tre / -3.0)
    # eh_norm = wp.length(eh)

    eF = wp.sqrt(wp.dot(eh, eh) + 1e-6)
    plastic_deformation = (
        eF + (3.0 * lam + 2.0 * mu) / (2.0 * mu) * tre * alpha - cohesion
    )

    if plastic_deformation <= 0:
        s_new = wp.vec3(wp.exp(e[0]), wp.exp(e[1]), wp.exp(e[2]))
        F_new = U * wp.diag(s) * wp.transpose(V)
        logdetF_new = e[0] + e[1] + e[2]  # log particle volume after projection

    if plastic_deformation > 0 and tre > 0:
        s_new = wp.vec3(1.0, 1.0, 1.0)
        F_new = U * wp.diag(s_new) * wp.transpose(V)
        logdetF_new = 0.0

    if plastic_deformation > 0 and tre <= 0:
        H = e - eh * (plastic_deformation / eF)
        s_new = wp.vec3(wp.exp(H[0]), wp.exp(H[1]), wp.exp(H[2]))
        F_new = U * wp.diag(s_new) * wp.transpose(V)
        logdetF_new = H[0] + H[1] + H[2]

    return F_new, logdetF_new - logdetF


@wp.kernel
def zero_everything(
    state: MPMStateStruct,
    ext_body_f: wp.array(dtype=wp.spatial_vector),
    int_body_f: wp.array(dtype=wp.spatial_vector),
    mpm_contact_count: wp.array(dtype=int),
    grid_dim_x: int,
    grid_dim_y: int,
    grid_dim_z: int,
    num_particles: int,
    num_bodies: int,
):
    tid = wp.tid()
    if tid == 0:
        state.grid_lower[0] = 2147483647
        state.grid_lower[1] = 2147483647
        state.grid_lower[2] = 2147483647

        state.grid_upper[0] = -2147483647
        state.grid_upper[1] = -2147483647
        state.grid_upper[2] = -2147483647

        mpm_contact_count[0] = 0
        state.error[0] = 0

    if tid < num_bodies:
        ext_body_f[tid] = wp.spatial_vector()
        int_body_f[tid] = wp.spatial_vector()

    grid_x = tid / grid_dim_z / grid_dim_y
    grid_y = (tid / grid_dim_z) % grid_dim_y
    grid_z = tid % grid_dim_z

    if tid < num_particles:
        state.particle_f[tid] = wp.vec3(0.0)

    if grid_x < grid_dim_x and grid_y < grid_dim_y and grid_z < grid_dim_z:
        if state.grid_m[grid_x, grid_y, grid_z] > 0.0:
            state.grid_m[grid_x, grid_y, grid_z] = 0.0
            state.grid_mv[grid_x, grid_y, grid_z] = wp.vec3(0.0)
            state.grid_v[grid_x, grid_y, grid_z] = wp.vec3(0.0)


@wp.kernel
def compute_grid_bound(model: MPMModelStruct, state: MPMStateStruct):
    tid = wp.tid()
    x = state.particle_q[tid]
    fx = (x[0] - model.dx * 4.0) * model.inv_dx
    fy = (x[1] - model.dx * 4.0) * model.inv_dx
    fz = (x[2] - model.dx * 4.0) * model.inv_dx
    ix = int(wp.floor(fx))
    iy = int(wp.floor(fy))
    iz = int(wp.floor(fz))
    wp.atomic_min(state.grid_lower, 0, ix)
    wp.atomic_min(state.grid_lower, 1, iy)
    wp.atomic_min(state.grid_lower, 2, iz)

    fx = (x[0] + model.dx * 4.0) * model.inv_dx
    fy = (x[1] + model.dx * 4.0) * model.inv_dx
    fz = (x[2] + model.dx * 4.0) * model.inv_dx
    ix = int(wp.ceil(fx))
    iy = int(wp.ceil(fy))
    iz = int(wp.ceil(fz))
    wp.atomic_max(state.grid_upper, 0, ix)
    wp.atomic_max(state.grid_upper, 1, iy)
    wp.atomic_max(state.grid_upper, 2, iz)


@wp.kernel
def set_grid_bound(model: MPMModelStruct, state: MPMStateStruct):
    state.grid_lower[0] = -model.grid_dim_x / 2
    state.grid_lower[1] = -model.grid_dim_y / 2
    # state.grid_lower[2] = -model.grid_dim_z / 2
    state.grid_lower[2] = -3

    state.grid_upper[0] = model.grid_dim_x / 2
    state.grid_upper[1] = model.grid_dim_y / 2
    state.grid_upper[2] = model.grid_dim_z - 4


@wp.kernel
def p2g(
    model: MPMModelStruct,
    state_in: MPMStateStruct,
    state_out: MPMStateStruct,
    gravity: wp.vec3,
    dt: float,
):
    p = wp.tid()
    contact_force = state_in.particle_f[p]
    x = state_in.particle_q[p]
    x = (
        x
        - wp.vec3(
            float(state_in.grid_lower[0]),
            float(state_in.grid_lower[1]),
            float(state_in.grid_lower[2]),
        )
        * model.dx
    )  # x relative to corner

    C = state_in.particle_C[p]
    mass = model.particle_mass[p]

    I33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    F_tmp = (I33 + state_in.particle_C[p] * dt) * state_in.particle_F[p]

    U, s, V = wp.svd3(F_tmp)

    new_v = state_in.particle_qd[p] + gravity * dt

    k_mu = model.particle_mu_lam_ys[p][0]
    k_lam = model.particle_mu_lam_ys[p][1]
    k_ys = model.particle_mu_lam_ys[p][2]
    vc = state_in.particle_volume_correction[p]

    type = model.particle_type[p]

    if type == 0:
        new_F, vc = compute_von_mises(F_tmp, U, s, V, k_ys, k_mu)

    if type == 1:
        k_friction = model.particle_friction_cohesion[p][0]
        k_cohesion = model.particle_friction_cohesion[p][1]
        new_F, delta_vc = compute_drucker_prager(
            F_tmp, U, s, V, k_mu, k_lam, k_friction, k_cohesion, vc
        )
        state_out.particle_volume_correction[p] = vc + delta_vc

    rest_volume = model.particle_vol[p]
    state_out.particle_F[p] = new_F

    if type == 2:
        current_volume = state_in.particle_vol[p]
        k_viscosity = model.particle_friction_cohesion[p][2]
        pressure = 1000.0 * ((rest_volume / current_volume) ** 7.0 - 1.0)
        pressure = wp.max(-10.0, pressure)
        c01 = (C[0, 1] + C[1, 0]) * k_viscosity
        c02 = (C[0, 2] + C[2, 0]) * k_viscosity
        c12 = (C[1, 2] + C[2, 1]) * k_viscosity
        stress_cauchy = wp.mat33(
            -pressure, c01, c02, c01, -pressure, c12, c02, c12, -pressure
        )

    else:
        J = wp.determinant(new_F)
        r = U * wp.transpose(V)
        stress_cauchy = (new_F - r) * wp.transpose(new_F) * (2.0 * k_mu) + I33 * (
            k_lam * J * (J - 1.0)
        )

    stress = stress_cauchy * (-4.0 * model.inv_dx * model.inv_dx * rest_volume * dt)
    affine = stress + C * mass
    mv = new_v * mass + contact_force * dt

    grid_pos = x * model.inv_dx
    base_pos_x = wp.int(grid_pos[0] - 0.5)
    base_pos_y = wp.int(grid_pos[1] - 0.5)
    base_pos_z = wp.int(grid_pos[2] - 0.5)
    fx = grid_pos - wp.vec3(
        wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
    )

    # https://dl.acm.org/doi/pwp/10.1145/2897826.2927348, Eqn. 123
    wa = wp.vec3(1.5) - fx
    wb = fx - wp.vec3(1.0)
    wc = fx - wp.vec3(0.5)
    w = wp.mat33(
        wp.cw_mul(wa, wa) * 0.5,
        wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
        wp.cw_mul(wc, wc) * 0.5,
    )

    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation

                dpos = (wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx) * model.dx
                momentum = mv + affine * dpos

                ix = base_pos_x + i
                iy = base_pos_y + j
                iz = base_pos_z + k

                wp.atomic_add(state_in.grid_mv, ix, iy, iz, momentum * weight)
                wp.atomic_add(state_in.grid_m, ix, iy, iz, mass * weight)


@wp.kernel
def g2p(
    model: MPMModelStruct,
    state_in: MPMStateStruct,
    state_out: MPMStateStruct,
    dt: float,
):
    p = wp.tid()
    lower = wp.vec3(
        float(state_in.grid_lower[0]) * model.dx,
        float(state_in.grid_lower[1]) * model.dx,
        float(state_in.grid_lower[2]) * model.dx,
    )
    x = state_in.particle_q[p] - lower

    base_pos_x = wp.int(x[0] * model.inv_dx - 0.5)
    base_pos_y = wp.int(x[1] * model.inv_dx - 0.5)
    base_pos_z = wp.int(x[2] * model.inv_dx - 0.5)

    fx = x * model.inv_dx - wp.vec3(
        wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
    )
    wa = wp.vec3(1.5) - fx
    wb = fx - wp.vec3(1.0)
    wc = fx - wp.vec3(0.5)
    w = wp.mat33(
        wp.cw_mul(wa, wa) * 0.5,
        wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
        wp.cw_mul(wc, wc) * 0.5,
    )

    new_v = wp.vec3(0.0, 0.0, 0.0)
    new_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    new_mass = float(0.0)
    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                weight = w[0, i] * w[1, j] * w[2, k]
                dpos = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx

                ix = base_pos_x + i
                iy = base_pos_y + j
                iz = base_pos_z + k
                v = state_in.grid_v[ix, iy, iz]

                new_v = new_v + v * weight
                new_C = new_C + wp.outer(v, dpos) * (weight * model.inv_dx * 4.0)
                new_mass += state_in.grid_m[ix, iy, iz] * weight

    new_volume = model.dx * model.dx * model.dx * model.particle_mass[p] / new_mass
    state_out.particle_vol[p] = new_volume

    # integration
    new_x = x + new_v * dt

    if wp.isnan(new_x[0]) or wp.isnan(new_x[1]) or wp.isnan(new_x[2]) or state_in.error[0] == 1:
        state_out.error[0] = 1
    else:
        state_out.error[0] = 0

    new_x = wp.vec3(
        wp.max(new_x[0], 3.0 * model.dx),
        wp.max(new_x[1], 3.0 * model.dx),
        wp.max(new_x[2], 3.0 * model.dx),
    )
    new_x = wp.vec3(
        wp.min(new_x[0], (float(model.grid_dim_x) - 3.0) * model.dx),
        wp.min(new_x[1], (float(model.grid_dim_y) - 3.0) * model.dx),
        wp.min(new_x[2], (float(model.grid_dim_z) - 3.0) * model.dx),
    )
    new_x = new_x + lower

    state_out.particle_q[p] = new_x
    state_out.particle_qd[p] = new_v
    state_out.particle_C[p] = new_C


@wp.func
def rigid_contact(
    x: wp.vec3,
    X_wb: wp.transform,
    X_bs: wp.transform,
    geo_type: int,
    geo_id: wp.uint64,
    geo_scale: wp.vec3,
    contact_margin: float,
):
    X_ws = wp.transform_multiply(X_wb, X_bs)
    X_sw = wp.transform_inverse(X_ws)

    # transform grid position to shape local space
    x_local = wp.transform_point(X_sw, x)

    # evaluate shape sdf
    d = 1.0e6
    n = wp.vec3()
    v = wp.vec3()

    # GEO_SPHERE (0)
    if geo_type == 0:
        d = sphere_sdf(wp.vec3(), geo_scale[0], x_local)
        n = sphere_sdf_grad(wp.vec3(), geo_scale[0], x_local)

    # GEO_BOX (1)
    if geo_type == 1:
        d = box_sdf(geo_scale, x_local)
        n = box_sdf_grad(geo_scale, x_local)

    # GEO_CAPSULE (2)
    if geo_type == 2:
        d = capsule_sdf(geo_scale[0], geo_scale[1], x_local)
        n = capsule_sdf_grad(geo_scale[0], geo_scale[1], x_local)

    # GEO_MESH (3)
    if geo_type == 3:
        mesh = geo_id

        face_index = int(0)
        face_u = float(0.0)
        face_v = float(0.0)
        sign = float(0.0)

        if wp.mesh_query_point(
            mesh,
            x_local / geo_scale[0],
            contact_margin,
            sign,
            face_index,
            face_u,
            face_v,
        ):

            shape_p = wp.mesh_eval_position(mesh, face_index, face_u, face_v)
            shape_v = wp.mesh_eval_velocity(mesh, face_index, face_u, face_v)

            shape_p = shape_p * geo_scale[0]

            delta = x_local - shape_p
            d = wp.length(delta) * sign
            n = wp.normalize(delta) * sign
            v = shape_v

    # GEO_DENSE_VOLUME
    if geo_type == 7:
        volume = geo_id
        p = x_local / geo_scale[0]
        nd = wp.dense_volume_sample_vec4(volume, p) * geo_scale[0]
        n = wp.normalize(wp.vec3(nd[0], nd[1], nd[2]))
        d = nd[3]

    body_pos = wp.vec3()
    body_vel = wp.vec3()
    world_normal = wp.vec3()
    if d < contact_margin:
        body_pos = wp.transform_point(X_bs, x_local - n * d)
        body_vel = wp.transform_vector(X_bs, v)
        world_normal = wp.transform_vector(X_ws, n)

    return d, body_pos, body_vel, world_normal


@wp.kernel
def grid_op(model: MPMModelStruct, state: MPMStateStruct, dt: float):
    tid = wp.tid()
    grid_x = tid / model.grid_dim_z / model.grid_dim_y
    grid_y = (tid / model.grid_dim_z) % model.grid_dim_y
    grid_z = tid % model.grid_dim_z
    m = state.grid_m[grid_x, grid_y, grid_z]
    if m > 1e-9:
        mv = state.grid_mv[grid_x, grid_y, grid_z]
        gv = mv / m

        lower = wp.vec3(
            float(state.grid_lower[0]) * model.dx,
            float(state.grid_lower[1]) * model.dx,
            float(state.grid_lower[2]) * model.dx,
        )
        gx = lower + wp.vec3(
            float(grid_x) * model.dx, float(grid_y) * model.dx, float(grid_z) * model.dx
        )

        # ground contact
        v_out = gv
        if wp.dot(gx, model.ground_normal) < 0.0001:
            if model.ground_sticky != 0:
                v_out = wp.vec3(0.0, 0.0, 0.0)
            else:
                v_n = wp.dot(v_out, model.ground_normal)
                v_t = v_out - model.ground_normal * v_n
                v_t_norm = wp.length(v_t) + 1e-6
                v_t_dir = v_t * (1.0 / v_t_norm)
                v_t = v_t_dir * wp.max(
                    v_t_norm + wp.min(v_n, 0.0) * model.static_mu, 0.0
                )
                v_out = v_t

        bound = 3
        if grid_x <= bound and v_out[0] < 0.0:
            v_out = wp.vec3(0.0, v_out[1], v_out[2])
        if grid_y <= bound and v_out[1] < 0.0:
            v_out = wp.vec3(v_out[0], 0.0, v_out[2])
        if grid_z <= bound and v_out[2] < 0.0:
            v_out = wp.vec3(v_out[0], v_out[1], 0.0)
        if grid_x >= model.grid_dim_x - bound and v_out[0] > 0.0:
            v_out = wp.vec3(0.0, v_out[1], v_out[2])
        if grid_y >= model.grid_dim_y - bound and v_out[1] > 0.0:
            v_out = wp.vec3(v_out[0], 0.0, v_out[2])
        if grid_z >= model.grid_dim_z - bound and v_out[2] > 0.0:
            v_out = wp.vec3(v_out[0], v_out[1], 0.0)

        state.grid_v[grid_x, grid_y, grid_z] = v_out


@wp.kernel
def grid_op_with_contact(
    model: MPMModelStruct,
    state: MPMStateStruct,
    dt: float,
    # body info
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    # shape info
    shape_X_bs: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_geo_type: wp.array(dtype=int),
    shape_geo_id: wp.array(dtype=wp.uint64),
    shape_geo_scale: wp.array(dtype=wp.vec3),
    num_shapes: int,
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    grid_x = tid / model.grid_dim_z / model.grid_dim_y
    grid_y = (tid / model.grid_dim_z) % model.grid_dim_y
    grid_z = tid % model.grid_dim_z
    m = state.grid_m[grid_x, grid_y, grid_z]
    if m > 1e-9:
        mv = state.grid_mv[grid_x, grid_y, grid_z]
        gv = mv / m

        lower = wp.vec3(
            float(state.grid_lower[0]) * model.dx,
            float(state.grid_lower[1]) * model.dx,
            float(state.grid_lower[2]) * model.dx,
        )
        gx = lower + wp.vec3(
            float(grid_x) * model.dx, float(grid_y) * model.dx, float(grid_z) * model.dx
        )

        v_in = gv
        v_out = gv

        for shape_index in range(num_shapes):
            rigid_index = shape_body[shape_index]
            X_wb = wp.transform_identity()
            body_v_s = wp.spatial_vector()
            X_com = wp.vec3()
            if rigid_index >= 0:
                X_wb = body_q[rigid_index]
                body_v_s = body_qd[rigid_index]
                X_com = body_com[rigid_index]

            X_bs = shape_X_bs[shape_index]
            geo_type = shape_geo_type[shape_index]
            geo_scale = shape_geo_scale[shape_index]
            geo_id = shape_geo_id[shape_index]

            contact_margin = model.particle_radius
            dist, body_pos, body_vel, world_normal = rigid_contact(
                gx, X_wb, X_bs, geo_type, geo_id, geo_scale, contact_margin
            )

            if dist < contact_margin:
                bx = wp.transform_point(X_wb, body_pos)
                r = bx - wp.transform_point(X_wb, X_com)

                body_w = wp.spatial_top(body_v_s)
                body_v = wp.spatial_bottom(body_v_s)
                bv = body_v + wp.cross(body_w, r) + wp.transform_vector(X_wb, body_vel)

                # TODO: influence = contact_margin - dist
                if model.body_sticky != 0:
                    v_out = bv
                else:
                    rel_v = v_in - bv
                    v_n = wp.dot(rel_v, world_normal)
                    v_t = rel_v - v_n * world_normal
                    v_t_norm = wp.length(v_t) + 1e-6
                    v_t_dir = v_t * (1.0 / v_t_norm)

                    mu = model.body_mu
                    v_t = v_t_dir * wp.max(v_t_norm + wp.min(v_n, 0.0) * mu, 0.0)
                    v_out = bv + v_t

                b_f = m * (v_in - v_out) * (1.0 / dt)
                b_t = wp.cross(r, b_f)

                # add force to body
                if rigid_index >= 0:
                    wp.atomic_add(body_f, rigid_index, wp.spatial_vector(b_t, b_f))

                v_in = v_out

        # ground contact
        if wp.dot(gx, model.ground_normal) < 0.0001:
            if model.ground_sticky != 0:
                v_out = wp.vec3(0.0, 0.0, 0.0)
            else:
                v_n = wp.dot(v_out, model.ground_normal)
                v_t = v_out - model.ground_normal * v_n
                v_t_norm = wp.length(v_t) + 1e-6
                v_t_dir = v_t * (1.0 / v_t_norm)
                v_t = v_t_dir * wp.max(v_t_norm + wp.min(v_n, 0.0) * model.static_mu, 0.0)
                v_out = v_t

        # boundary condition, we hope they are never used
        bound = 3
        if grid_x <= bound and v_out[0] < 0.0:
            v_out = wp.vec3(0.0, v_out[1], v_out[2])
        if grid_y <= bound and v_out[1] < 0.0:
            v_out = wp.vec3(v_out[0], 0.0, v_out[2])
        if grid_z <= bound and v_out[2] < 0.0:
            v_out = wp.vec3(v_out[0], v_out[1], 0.0)
        if grid_x >= model.grid_dim_x - bound and v_out[0] > 0.0:
            v_out = wp.vec3(0.0, v_out[1], v_out[2])
        if grid_y >= model.grid_dim_y - bound and v_out[1] > 0.0:
            v_out = wp.vec3(v_out[0], 0.0, v_out[2])
        if grid_z >= model.grid_dim_z - bound and v_out[2] > 0.0:
            v_out = wp.vec3(v_out[0], v_out[1], 0.0)

        state.grid_v[grid_x, grid_y, grid_z] = v_out


@wp.kernel
def create_soft_contacts(
    num_particles: int,
    particle_x: wp.array(dtype=wp.vec3),
    body_X_wb: wp.array(dtype=wp.transform),
    shape_X_bs: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_geo_type: wp.array(dtype=int),
    shape_geo_id: wp.array(dtype=wp.uint64),
    shape_geo_scale: wp.array(dtype=wp.vec3),
    soft_contact_margin: float,
    # outputs,
    soft_contact_count: wp.array(dtype=int),
    soft_contact_particle: wp.array(dtype=int),
    soft_contact_body: wp.array(dtype=int),
    soft_contact_body_pos: wp.array(dtype=wp.vec3),
    soft_contact_body_vel: wp.array(dtype=wp.vec3),
    soft_contact_normal: wp.array(dtype=wp.vec3),
    soft_contact_max: int,
):

    tid = wp.tid()

    shape_index = tid // num_particles  # which shape
    particle_index = tid % num_particles  # which particle
    rigid_index = shape_body[shape_index]

    px = particle_x[particle_index]

    X_wb = wp.transform_identity()
    if rigid_index >= 0:
        X_wb = body_X_wb[rigid_index]

    X_co = shape_X_bs[shape_index]

    X_so = wp.transform_multiply(X_wb, X_co)
    X_os = wp.transform_inverse(X_so)

    # transform particle position to shape local space
    x_local = wp.transform_point(X_os, px)

    # geo description
    geo_type = shape_geo_type[shape_index]
    geo_scale = shape_geo_scale[shape_index]

    # evaluate shape sdf
    d = 1.0e6
    n = wp.vec3()
    v = wp.vec3()

    # GEO_SPHERE (0)
    if geo_type == 0:
        d = sphere_sdf(wp.vec3(), geo_scale[0], x_local)
        n = sphere_sdf_grad(wp.vec3(), geo_scale[0], x_local)

    # GEO_BOX (1)
    if geo_type == 1:
        d = box_sdf(geo_scale, x_local)
        n = box_sdf_grad(geo_scale, x_local)

    # GEO_CAPSULE (2)
    if geo_type == 2:
        d = capsule_sdf(geo_scale[0], geo_scale[1], x_local)
        n = capsule_sdf_grad(geo_scale[0], geo_scale[1], x_local)

    # GEO_MESH (3)
    if geo_type == 3:
        mesh = shape_geo_id[shape_index]

        face_index = int(0)
        face_u = float(0.0)
        face_v = float(0.0)
        sign = float(0.0)

        if wp.mesh_query_point(
            mesh,
            x_local / geo_scale[0],
            soft_contact_margin,
            sign,
            face_index,
            face_u,
            face_v,
        ):

            shape_p = wp.mesh_eval_position(mesh, face_index, face_u, face_v)
            shape_v = wp.mesh_eval_velocity(mesh, face_index, face_u, face_v)

            shape_p = shape_p * geo_scale[0]

            delta = x_local - shape_p
            d = wp.length(delta) * sign
            n = wp.normalize(delta) * sign
            v = shape_v

    # GEO_DENSE_VOLUME
    if geo_type == 7:
        volume = shape_geo_id[shape_index]
        p = x_local / geo_scale[0]
        nd = wp.dense_volume_sample_vec4(volume, p) * geo_scale[0]
        n = wp.normalize(wp.vec3(nd[0], nd[1], nd[2]))
        d = nd[3]

    if d < soft_contact_margin:

        index = wp.atomic_add(soft_contact_count, 0, 1)

        if index < soft_contact_max:

            # compute contact point in body local space
            body_pos = wp.transform_point(X_co, x_local - n * d)
            body_vel = wp.transform_vector(X_co, v)

            world_normal = wp.transform_vector(X_so, n)

            soft_contact_body[index] = rigid_index
            soft_contact_body_pos[index] = body_pos
            soft_contact_body_vel[index] = body_vel
            soft_contact_particle[index] = particle_index
            soft_contact_normal[index] = world_normal


@wp.kernel
def eval_soft_contacts(
    model: MPMModelStruct,
    state: MPMStateStruct,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    contact_count: wp.array(dtype=int),
    contact_particle: wp.array(dtype=int),
    contact_body: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_distance: float,
    # outputs
    body_f: wp.array(dtype=wp.spatial_vector),
):

    tid = wp.tid()

    count = contact_count[0]
    if tid >= count:
        return

    rigid_index = contact_body[tid]
    particle_index = contact_particle[tid]

    px = state.particle_q[particle_index]
    pv = state.particle_qd[particle_index]

    X_wb = wp.transform_identity()
    X_com = wp.vec3()

    if rigid_index >= 0:
        X_wb = body_q[rigid_index]
        X_com = body_com[rigid_index]

    # body position in world space
    bx = wp.transform_point(X_wb, contact_body_pos[tid])
    r = bx - wp.transform_point(X_wb, X_com)

    n = contact_normal[tid]
    c = wp.dot(n, px - bx) - contact_distance

    if c > 0.0:
        return

    # body velocity
    body_v_s = wp.spatial_vector()
    if rigid_index >= 0:
        body_v_s = body_qd[rigid_index]

    body_w = wp.spatial_top(body_v_s)
    body_v = wp.spatial_bottom(body_v_s)

    # compute the body velocity at the particle position
    bv = body_v + wp.cross(body_w, r) + wp.transform_vector(X_wb, contact_body_vel[tid])

    # relative velocity
    v = pv - bv

    # decompose relative velocity
    vn = wp.dot(n, v)
    vt = v - n * vn

    # contact elastic
    fn = n * c * model.body_ke

    # contact damping
    fd = n * wp.min(vn, 0.0) * model.body_kd

    kf = 1.0

    # Coulomb friction (smooth, but gradients are numerically unstable around |vt| = 0)
    ft = wp.normalize(vt) * wp.min(kf * wp.length(vt), abs(model.body_mu * c * model.body_ke))

    f_total = fn + (fd + ft)
    t_total = wp.cross(r, f_total)

    wp.atomic_sub(state.particle_f, particle_index, f_total)

    if rigid_index >= 0:
        wp.atomic_add(body_f, rigid_index, wp.spatial_vector(t_total, f_total))
