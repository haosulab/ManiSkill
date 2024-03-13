import sapien

sapien.set_log_level("info")

sapien.physx.enable_gpu()
sapien.render.set_viewer_shader_dir("../vulkan_shader/default")

sapien.render.set_camera_shader_dir("../vulkan_shader/minimal")
sapien.render.set_picture_format("Color", "r8g8b8a8unorm")
sapien.render.set_picture_format("ColorRaw", "r8g8b8a8unorm")
sapien.render.set_picture_format("PositionSegmentation", "r16g16b16a16sint")


def main():
    scene_count = 256

    scenes: list[sapien.Scene] = []

    px = sapien.physx.PhysxGpuSystem()

    for i in range(scene_count):
        scene = sapien.Scene([px, sapien.render.RenderSystem()])
        px.set_scene_offset(scene, [i * 10.0, 0.0, 0.0])
        scenes.append(scene)

    from sapien.wrapper.urdf_loader import URDFLoader

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

    [link for robot in robots for link in robot.links]

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
    for b in balls:
        print(
            "ball {:x}".format(
                b.find_component_by_type(
                    sapien.physx.PhysxRigidDynamicComponent
                )._physx_pointer
            )
        )

    cams = []
    for scene, robot in zip(scenes, robots):
        cam = scene.add_mounted_camera(
            "", robot.links[6].entity, sapien.Pose([0.5, 0, 0]), 256, 512, 1, 0.01, 10
        )
        cams.append(cam)

    px.gpu_init()
    px.step()
    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":

    for _ in range(100):
        main()
