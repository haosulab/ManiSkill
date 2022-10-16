import warp as wp
import numpy as np


@wp.kernel
def rasterize_clear_kernel(output: wp.array(dtype=int, ndim=2), value: int):
    h, w = wp.tid()
    output[h, w] = value


@wp.kernel
def rasterize_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    offset: wp.vec3,  # world space, e.g. canvas_size / xy_scale
    xy_scale: float,  # e.g. canvas_size / world_size
    z_scale: float,  # z scale
    radius: int,  # radius in screen space, point_radius * xy_scale
    canvas_width: int,
    canvas_height: int,
    canvas: wp.array(dtype=int, ndim=2),
):
    tid = wp.tid()
    q = particle_q[tid] + offset

    x_center = q[0]
    y_center = q[1]
    z_center = q[2]

    sx_center = x_center * xy_scale
    sy_center = y_center * xy_scale

    sx_start = int(wp.round(sx_center - float(radius)))
    sy_start = int(wp.round(sy_center - float(radius)))

    r = float(radius) / xy_scale  # radius in world space
    r2 = r * r

    for i in range(radius * 2 + 1):
        for j in range(radius * 2 + 1):
            grid_x = sx_start + i
            grid_y = sy_start + j

            if (
                grid_x >= 0
                and grid_x < canvas_width
                and grid_y >= 0
                and grid_y < canvas_height
            ):
                x = float(grid_x) / xy_scale - x_center
                y = float(grid_y) / xy_scale - y_center

                z2 = r2 - x * x - y * y
                if z2 > 0:
                    z = wp.sqrt(z2) + z_center
                    wp.atomic_max(canvas, grid_y, grid_x, int(z * z_scale))


if __name__ == "__main__":
    wp.init()
    width = 200
    height = 100

    canvas = wp.zeros((height, width), dtype=wp.int32, device="cuda")
    particles = wp.array(
        np.array([[0, 0, 0]], dtype=np.float32), dtype=wp.vec3, device="cuda"
    )

    wp.launch(
        rasterize_clear_kernel, dim=(height, width), inputs=[canvas, 0], device="cuda"
    )

    wp.launch(
        rasterize_kernel,
        dim=particles.shape[0],
        inputs=[
            particles,
            wp.vec3(10.0, 5.0, 0.0),
            10,  # xy_scale
            1000,  # z_scale
            10,  # radius
            width,
            height,
            canvas,
        ],
        device="cuda",
    )

    wp.synchronize()

    arr = canvas.numpy().reshape((height, width))

    import matplotlib.pyplot as plt

    plt.imshow(arr)
    plt.show()
