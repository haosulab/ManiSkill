import sapien

def set_simulation_quality(render_quality):
    ENABLE_SHADOWS = True

    if render_quality == "high":
        print("High quality raycasting is set")
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_viewer_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(64)
        sapien.render.set_ray_tracing_path_depth(16)
        sapien.render.set_ray_tracing_denoiser("optix")
        ENABLE_SHADOWS = True

    elif render_quality == "medium":
        print("Medium quality raycasting is set")
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_viewer_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(4)
        sapien.render.set_ray_tracing_path_depth(3)
        sapien.render.set_ray_tracing_denoiser("optix")
        ENABLE_SHADOWS = False

    elif render_quality == "low":
        print("Low quality raycasting is set")
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_viewer_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(2)
        sapien.render.set_ray_tracing_path_depth(1)
        sapien.render.set_ray_tracing_denoiser("none")
        ENABLE_SHADOWS = False

    else: # rasterization
        print("Rasterization is set")
        ENABLE_SHADOWS = False

    return ENABLE_SHADOWS