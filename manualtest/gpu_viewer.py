import numpy as np
import sapien
from sapien.asset import create_dome_envmap
from sapien.utils.viewer.viewer import ControlWindow, Viewer

sapien.physx.enable_gpu()
# sapien.set_cuda_tensor_backend("torch")

sapien.render.set_viewer_shader_dir("../vulkan_shader/default")

sapien.render.set_camera_shader_dir("../vulkan_shader/minimal")
sapien.render.set_picture_format("Color", "r8g8b8a8unorm")
sapien.render.set_picture_format("ColorRaw", "r8g8b8a8unorm")
sapien.render.set_picture_format("outPositionSegmentation", "r16g16b16a16sint")


def main():
    scene_count = 64

    scenes = []
    for i in range(scene_count):
        scene = sapien.Scene()
        px: sapien.physx.PhysxGpuSystem = scene.physx_system
        px.set_scene_offset(scene, [i * 10, 0, 0])
        scenes.append(scene)

    px: sapien.physx.PhysxGpuSystem = scenes[0].physx_system

    urdf_loader = scenes[0].create_urdf_loader()
    builder = urdf_loader.load_file_as_articulation_builder(
        "./mani_skill /assets/robots/panda/panda_v2.urdf"
    )
    robots = []
    for scene in scenes:
        scene.load_widget_from_package("demo_arena", "DemoArena")
        builder.set_scene(scene)
        robot = builder.build()
        robots.append(robot)

    all_links = [link for robot in robots for link in robot.links]

    builder = scenes[0].create_actor_builder()
    builder.add_sphere_collision(radius=0.1)
    builder.add_sphere_visual(radius=0.1)

    balls = []
    z = 5
    for scene in scenes:
        builder.set_scene(scene)
        builder.set_initial_pose(sapien.Pose([0, 0, z]))
        ball = builder.build()
        balls.append(ball)
        z += 1

    all_bodies = [
        b.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent) for b in balls
    ]

    cams = []
    for scene, robot in zip(scenes, robots):
        cam = scene.add_mounted_camera(
            "", robot.links[6].entity, sapien.Pose([0.5, 0, 0]), 256, 512, 1, 0.01, 10
        )
        # cam = scene.add_camera("", 256, 512, 1, 0.01, 10)
        # cam.entity.set_pose(sapien.Pose([-2, 0, 0.5]))
        cams.append(cam)

    px.gpu_init()

    import matplotlib.pyplot as plt

    px.gpu_fetch_articulation_link_pose()

    def fast_way():
        # tell render shapes where to look for poses
        for body in all_links + all_bodies:
            rb = body.entity.find_component_by_type(sapien.render.RenderBodyComponent)
            if rb is None:
                continue
            for s in rb.render_shapes:
                s.set_gpu_pose_batch_index(body.gpu_pose_index)

        for cam in cams:
            body = cam.entity.find_component_by_type(
                sapien.physx.PhysxRigidBodyComponent
            )
            if body is None:
                continue
            cam.set_gpu_pose_batch_index(body.gpu_pose_index)

        # render system group manages batched rendering
        render_system_group = sapien.render.RenderSystemGroup(
            [s.render_system for s in scenes]
        )

        # camera group renders images in batches
        camera_group = render_system_group.create_camera_group(
            cams, ["Color", "PositionSegmentation"]
        )
        render_system_group.set_cuda_poses(px.cuda_rigid_body_data)

        px.gpu_fetch_rigid_dynamic_data()
        px.gpu_fetch_articulation_link_pose()
        render_system_group.update_render()

        for _ in range(10):
            for _ in range(20):
                px.step()

            px.gpu_fetch_rigid_dynamic_data()
            px.gpu_fetch_articulation_link_pose()

            render_system_group.update_render()
            camera_group.take_picture()
            color = camera_group.get_picture_cuda("Color")
            color = color.cpu().numpy()

            ps = camera_group.get_picture_cuda("PositionSegmentation")
            ps = ps.cpu().numpy()

            plt.subplot(2, 2, 1)
            plt.imshow(color[0][..., :3])
            plt.subplot(2, 2, 2)
            plt.imshow(color[1][..., :3])

            plt.subplot(2, 2, 3)
            plt.imshow((ps[0][..., :3] + 32768) / 65535)
            plt.subplot(2, 2, 4)
            plt.imshow((ps[1][..., :3] + 32768) / 65535)

            plt.show()

    def slow_way():
        for _ in range(10):
            for _ in range(20):
                px.step()

            px.sync_poses_gpu_to_cpu()
            for scene in scenes:
                scene.update_render()
            for cam in cams:
                cam.take_picture()

            pictures = [cam.get_picture("Color") for cam in cams]

            plt.subplot(1, 2, 1)
            plt.imshow(pictures[0])
            plt.subplot(1, 2, 2)
            plt.imshow(pictures[1])
            plt.show()

    # slow_way()
    # fast_way()

    viewer = Viewer()
    viewer.set_scene(scenes[0])
    # viewer.set_scenes(scenes)
    vs = viewer.window._internal_scene
    vs.set_ambient_light([0.1, 0.1, 0.1])
    vs.set_cubemap(scenes[0].render_system.get_cubemap()._internal_cubemap)

    viewer.render()

    while not viewer.closed:
        px.step()
        px.sync_poses_gpu_to_cpu()
        viewer.window.update_render()
        viewer.render()


main()
