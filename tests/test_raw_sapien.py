"""
Minimal code for very high performance sim and rendering with SAPIEN without ManiSkill framework
"""

import time
from dataclasses import dataclass

import numpy as np
import sapien
import sapien.physx as physx
import torch
import tyro

# import torch
from line_profiler import profile

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.envs.utils.system.backend import parse_sim_and_render_backend
from mani_skill.render.utils import can_render
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@dataclass
class Args:
    num_envs: int = 16
    sim_backend: str = "physx_cuda"
    render_backend: str = "cuda"
    scene_preset: str = "franka_floor"
    num_steps: int = 100
    step_physics: bool = True
    random_actions: bool = True
    fetch_observations: bool = True


# sapien.set_log_level("info")
# sapien.render.set_log_level("info")


def set_sim_config(sim_config: SimConfig):
    physx.set_shape_config(
        contact_offset=sim_config.scene_config.contact_offset,
        rest_offset=sim_config.scene_config.rest_offset,
    )
    physx.set_body_config(
        solver_position_iterations=sim_config.scene_config.solver_position_iterations,
        solver_velocity_iterations=sim_config.scene_config.solver_velocity_iterations,
        sleep_threshold=sim_config.scene_config.sleep_threshold,
    )
    physx.set_scene_config(
        gravity=sim_config.scene_config.gravity,
        bounce_threshold=sim_config.scene_config.bounce_threshold,
        enable_pcm=sim_config.scene_config.enable_pcm,
        enable_tgs=sim_config.scene_config.enable_tgs,
        enable_ccd=sim_config.scene_config.enable_ccd,
        enable_enhanced_determinism=sim_config.scene_config.enable_enhanced_determinism,
        enable_friction_every_iteration=sim_config.scene_config.enable_friction_every_iteration,
        cpu_workers=sim_config.scene_config.cpu_workers,
    )
    physx.set_default_material(**sim_config.default_materials_config.dict())


def setup_sapien_scenes(num_envs: int, sim_config: SimConfig):
    backend = parse_sim_and_render_backend("physx_cuda", "cuda")
    physx_system = sapien.physx.PhysxGpuSystem()

    sub_scenes = []
    scene_grid_length = int(np.ceil(np.sqrt(num_envs)))
    for scene_idx in range(num_envs):
        scene_x, scene_y = (
            scene_idx % scene_grid_length - scene_grid_length // 2,
            scene_idx // scene_grid_length - scene_grid_length // 2,
        )
        systems = [physx_system]
        if can_render(backend.render_device):
            systems.append(sapien.render.RenderSystem(backend.render_device))
        scene = sapien.Scene(systems=systems)
        physx_system.set_scene_offset(
            scene,
            [
                scene_x * sim_config.spacing,
                scene_y * sim_config.spacing,
                0,
            ],
        )
        sub_scenes.append(scene)

    assert sub_scenes[0].physx_system == sub_scenes[1].physx_system

    return sub_scenes


def create_franka_floor_scene(sub_scenes: list[sapien.Scene]):
    loader = sub_scenes[0].create_urdf_loader()
    builder = loader.load_file_as_articulation_builder(
        str(PACKAGE_ASSET_DIR / "robots" / "panda" / "panda_v2.urdf")
    )

    for scene in sub_scenes:
        builder.set_scene(scene)
        builder.initial_pose = sapien.Pose([0, 0, 1], [1, 0, 0, 0])
        builder.build()

    altitude = 0
    render_half_size = [10, 10]
    render_material = sapien.render.RenderMaterial(base_color=[0.8, 0.8, 0.8, 1])

    builder = sub_scenes[0].create_actor_builder()
    builder.set_physx_body_type("static")
    builder.add_plane_visual(
        sapien.Pose(p=[0, 0, altitude], q=[0.7071068, 0, -0.7071068, 0]),
        [10, *render_half_size],
        render_material,
        "",
    )
    for scene in sub_scenes[1:]:
        builder.set_scene(scene)
        builder.build()
    builder.add_plane_collision(
        sapien.Pose(p=[0, 0, altitude], q=[0.7071068, 0, -0.7071068, 0]),
    )
    builder.set_scene(sub_scenes[0])
    builder.build()

    builder = sub_scenes[0].create_actor_builder()
    builder.add_box_visual(half_size=[0.02, 0.02, 0.02])
    builder.add_box_collision(half_size=[0.02, 0.02, 0.02])
    builder.set_initial_pose(sapien.Pose([0.3, 0, 0.02], [1, 0, 0, 0]))
    for scene in sub_scenes:
        builder.set_scene(scene)
        builder.build()

    # lighting
    for scene in sub_scenes:
        # scene.add_ground(altitude=0, render=True, render_material=sapien.render.RenderMaterial(base_color=[0.8, 0.8, 0.8, 1]))
        scene.set_ambient_light([0.3, 0.3, 0.3])


@profile
def main(args: Args):
    # initialize the scenes
    sim_config = SimConfig(
        sim_freq=100, control_freq=100, gpu_memory_config=GPUMemoryConfig()
    )
    sapien.physx.enable_gpu()
    sub_scenes = setup_sapien_scenes(args.num_envs, sim_config)
    px: sapien.physx.PhysxGpuSystem = sub_scenes[
        0
    ].physx_system  # the physx system, which is where you can access all simulation data

    physics_steps_per_control_step = sim_config.sim_freq // sim_config.control_freq

    print("==== Sim Config ====")
    print(
        f"sim_freq: {sim_config.sim_freq}, control_freq: {sim_config.control_freq}, num_envs: {args.num_envs}, num_steps: {args.num_steps}"
    )
    print("==== Sim Config ====")

    create_franka_floor_scene(sub_scenes)

    px.gpu_init()  # NOTE this will take one physics step

    start_time = time.perf_counter()

    @profile
    def loop():
        for _ in range(args.num_steps):
            # TODO (stao): take action
            if args.random_actions:
                action = torch.rand(size=(args.num_envs, 9))
                px.cuda_articulation_target_qpos.torch()[:] = action
                px.gpu_apply_articulation_target_position()

            for _ in range(physics_steps_per_control_step):
                if args.step_physics:
                    px.step()
            if args.fetch_observations:
                px.gpu_fetch_articulation_qpos()
                px.gpu_fetch_articulation_qvel()
                px.gpu_fetch_rigid_dynamic_data()
                px.cuda_articulation_qpos.torch()
                px.cuda_articulation_qvel.torch()
                px.cuda_rigid_body_data.torch()
                # torch.cuda.synchronize()
                # import ipdb; ipdb.set_trace()

    loop()
    end_time = time.perf_counter()
    frames = args.num_steps * args.num_envs
    print(f"FPS: {frames / (end_time - start_time):,.2f}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
