import numpy as np
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

    elif render_quality == "ultra":
        print("Ultra high quality raycasting is set")
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_viewer_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(256)
        sapien.render.set_ray_tracing_path_depth(32)
        sapien.render.set_ray_tracing_denoiser("optix")
        ENABLE_SHADOWS = True

    else: # rasterization
        print("Rasterization is set")
        sapien.render.set_camera_shader_dir("")
        sapien.render.set_viewer_shader_dir("")
        sapien.render.set_ray_tracing_denoiser("none")
        ENABLE_SHADOWS = False

    return ENABLE_SHADOWS

# Pre-determined sample quantities:
class SimulationQuantities:
    RESOLUTIONS = [(128, 128), (256, 256), (512, 512), (1024, 1024), (224, 224)]

    LIGHT_COLORS = [
        np.array([ 0.5, 0.5, 0.5 ]),
        np.array([ 1.0, 0.5, 0.5 ]),
        np.array([ 0.5, 1.0, 0.5 ]),
        np.array([ 0.5, 0.5, 1.0 ]),
    ]
    
    LIGHT_DIRECTIONS = [
        np.array([ [0, 1, -1] ]),
        np.array([ [0, 1.2, -1] ]),
        np.array([ [0.5, 1, -1.3] ]),
        np.array([ [0.5, 1.2, -1] ]),
    ]

    SPECULARITY = [0.0, 0.5, 1.0] # 0.0 is mirror-like

    METALLICITY = [0.0, 0.5, 1.0] # 0.0 is for non-metals and 1.0 is for metals
    
    INDEX_OF_REFRACTION = [1.0, 1.4500000476837158, 1.9]
    
    TRANSMISSION = [0.0, 0.5, 1.0]
    
    MATERIAL_COLORS = [
        # R,G,B,A
        np.array([12, 42, 160, 255]), 
        np.array([160, 42, 12, 255]), 
        np.array([12, 160, 42, 255]), 
        np.array([12, 42, 160, 100]), 
        np.array([12, 42, 160, 30]), 
    ]

def randomize_quantity(low=0, src: list=[]):
    random_idx = int(np.random.uniform(low=low, high=len(src) - 1))
    random_quantity = src[random_idx]
    print(f"Chosen quantity: {random_quantity}")
    return random_quantity
