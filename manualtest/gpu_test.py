import torch
import sapien
import sapien.physx
from sapien.utils import Viewer
import numpy as np
if __name__ == "__main__":
    torch.set_printoptions(precision=3, sci_mode=False)
    sapien.physx.enable_gpu()
    sapien.set_cuda_tensor_backend("torch")


    def create_scene(offset):
        scene = sapien.Scene()
        scene.physx_system.set_scene_offset(scene, offset)
        scene.set_ambient_light([0.5, 0.5, 0.5])
        return scene


    scene0 = create_scene([0, 0, 0])
    scene1 = create_scene([10, 0, 0])

    assert scene0.physx_system == scene1.physx_system
    physx_system: sapien.physx.PhysxGpuSystem = scene0.physx_system

    
    from mani_skill2.envs.scene import ManiSkillScene
    scene = ManiSkillScene([scene0, scene1], debug_mode=True)
    # scene.sub_scenes = [scene0, scene1]

    builder = scene.create_actor_builder()
    builder.add_box_visual(half_size=(0.25, 0.25, 0.25), material=[1, 0,0, 1])
    builder.add_box_collision(half_size=(0.25, 0.25, 0.25))
    builder.initial_pose = sapien.Pose(p=[2,0,1])
    actor1 = builder.build(name="cube-0")

    builder = scene.create_actor_builder()
    builder.add_box_visual(half_size=(0.25, 0.25, 0.25), material=[1, 0,0, 1])
    builder.add_box_collision(half_size=(0.25, 0.25, 0.25))
    builder.initial_pose = sapien.Pose(p=[5, 5, 5])
    actor2 = builder.build(name="cube-1")
    
    # physx_system.gpu_init()
    # scene._setup_gpu()
    # # loader = scene.create_urdf_loader()
    # # loader.name = "panda"
    # # builder = loader.load_file_as_articulation_builder("./mani_skill2/assets/robots/panda/panda_v2.urdf")
    # # panda = builder.build()
    # rigid_body_components = []
    # for ar in [actor1, actor2]:
    #     rigid_body_components += [e.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent) for e in ar._entities]
    # # # scene._body_index_buffer = physx_system.gpu_create_body_index_buffer(comps)
    # data_buffer_raw = physx_system.gpu_create_body_data_buffer(len(rigid_body_components))
    # data_buffer = scene._body_data_buffer

    # index_buffer = scene._body_index_buffer#physx_system.gpu_create_body_index_buffer(rigid_body_components)
    # offset_buffer = scene._body_offset_buffer# physx_system.gpu_create_body_offset_buffer(rigid_body_components)
    # # physx_system.gpu_query_body_data(data_buffer, index_buffer, offset_buffer)
    # physx_system._gpu_query_body_data_raw(data_buffer_raw, index_buffer)
    # # scene._update_gpu_buffers()
    # print("cube0", "pos", data_buffer[:2, :3], "vel", data_buffer[:2, 7:10])
    # print("cube0-raw", "pos", data_buffer_raw[:2, 4:7], "vel", data_buffer[:2, 8:11])
    # print("cube1", "pos", data_buffer[2:4, :3], "vel", data_buffer[2:4, 7:10])
    # print("cube1-raw", "pos", data_buffer_raw[2:4, 4:7], "vel", data_buffer[2:4, 8:11])
    # import ipdb;ipdb.set_trace()
    # for i, ar in enumerate([actor1, actor2]):
    #     ar._body_data_index_ref = scene._body_index_buffer
    #     ar._data_index = slice(2*i,2*i+2)
    scene._setup_gpu()
    # scene._update_gpu_buffers()
    print("cube0", scene.actors["cube-0"].get_raw_data())

    print("cube1", scene.actors["cube-1"].get_raw_data())
    # cube = scene.actors["cube-0"]
    print("cube0", scene.actors["cube-0"].pose.p)
    print("cube1", scene.actors["cube-1"].pose.p)
    
    # pose = cube.pose
    # # pose.p = [5, 5, 23]
    # cube.set_pose(sapien.Pose(p=[5,5,23]))
    # cube.set_linear_velocity([0, 0, 0])
    # cube.set_angular_velocity([0, 0, 0])
    # scene._update_gpu_buffers()
    # print("cube0", scene.actors["cube-0"].pose.p)
    # print("cube1", scene.actors["cube-1"].pose.p)
    # for i in range(20):
    #     physx_system.step()
    # scene._update_gpu_buffers() # after a physx_system.step, must always call this to query all gpu data
    # print("cube0", scene.actors["cube-0"].pose.p)
    # print("cube1", scene.actors["cube-1"].pose.p)
    
    import ipdb;ipdb.set_trace()

    # root_pose_buffer[0] = torch.tensor([0.1, 0, 0, 0, 1, 0, 0])
    # root_pose_buffer[1] = torch.tensor([0, 0.1, 0, 0, 0, 1, 0])

    # physx_system.gpu_apply_articulation_root_pose(root_pose_buffer, ai, offset_buffer)
    # root_pose_buffer[0] = torch.tensor([0, 0, 0, 0, 0, 0, 0])
    # root_pose_buffer[1] = torch.tensor([0, 0, 0, 0, 0, 0, 0])

    # # physx_system.gpu_update_articulation_kinematics()                                                                                                                                                                                                                         

    # physx_system.gpu_query_articulation_root_pose(root_pose_buffer, ai, offset_buffer)
    # print("sapien pose", root_pose_buffer)

    # physx_system._gpu_query_articulation_root_pose_raw(root_pose_buffer, ai)
    # print("raw pose", root_pose_buffer)

    # from mani_skill2.utils.structs.pose import Pose
    # import ipdb;ipdb.set_trace()
    # root_pose = Pose.create(root_pose_buffer)
    # renderer = sapien.SapienRenderer()
    # viewer = Viewer(renderer)
    # viewer.set_scene()
    # viewer.paused = True
    # # import ipdb;ipdb.set_trace()
    # root_pose = root_pose_buffer[0].cpu().numpy()
    # # r0.links[0].entity.set_pose(sapien.Pose(p=root_pose[:3], q=root_pose[3:]))
    # while True:
    #     viewer.render()