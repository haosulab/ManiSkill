import sapien
import sapien.physx
import torch

from mani_skill2.utils.structs.pose import vectorize_pose

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

builder = scene0.create_actor_builder()
builder.add_box_visual(half_size=(0.1, 0.1, 0.1), material=[1, 0, 0, 1])
builder.add_box_collision(half_size=(0.1, 0.1, 0.1))
builder.initial_pose = sapien.Pose(p=[0, 0, 1])

builder.set_scene(scene0)
cube00 = builder.build()
builder.set_scene(scene1)
cube01 = builder.build()

builder = scene0.create_actor_builder()
builder.add_box_visual(half_size=(0.25, 0.25, 0.25), material=[1, 0, 0, 1])
builder.add_box_collision(half_size=(0.25, 0.25, 0.25))
builder.initial_pose = sapien.Pose(p=[5, 5, 2])
builder.set_scene(scene0)
cube10 = builder.build()
builder.set_scene(scene1)
cube11 = builder.build()

physx_system.gpu_init()

torch.set_printoptions(precision=3, sci_mode=False)
actors = [cube00, cube01, cube10, cube11]
rigid_body_components = [
    a.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent) for a in actors
]

data_buffer_raw = physx_system.gpu_create_body_data_buffer(len(rigid_body_components))

data_buffer = physx_system.gpu_create_body_data_buffer(len(rigid_body_components))


index_buffer = physx_system.gpu_create_body_index_buffer(rigid_body_components)
offset_buffer = physx_system.gpu_create_body_offset_buffer(rigid_body_components)
print(physx_system.timestep)
physx_system.gpu_query_body_data(data_buffer, index_buffer, offset_buffer)
physx_system._gpu_query_body_data_raw(data_buffer_raw, index_buffer)
print("cube0", "pos", data_buffer[:2, :3], "vel", data_buffer[:2, 7:10])
print("cube0-raw", "pos", data_buffer_raw[:2, 4:7], "vel", data_buffer[:2, 8:11])

# # print("cube1", data_buffer[2:4, :3])

data_buffer[..., :7] = torch.tensor([0, 0, 1, 1, 0, 0, 0])
data_buffer[..., 7:] = 0
physx_system.gpu_apply_body_data(data_buffer, index_buffer, offset_buffer)

# physx_system.gpu_apply_body_data(data_buffer[:2], index_buffer[:2], offset_buffer[:2])

physx_system.gpu_query_body_data(data_buffer, index_buffer, offset_buffer)
physx_system._gpu_query_body_data_raw(data_buffer_raw, index_buffer)
print("cube0-raw", "pos", data_buffer_raw[:2, 4:7], "vel", data_buffer[:2, 8:11])
print("cube1-raw", "pos", data_buffer_raw[2:4, 4:7], "vel", data_buffer[2:4, 8:11])


# loader = scene0.create_urdf_loader()
# builder = loader.load_file_as_articulation_builder("./mani_skill2/assets/robots/panda/panda_v2.urdf")

# builder.set_scene(scene0)
# r0 = builder.build()

# builder.set_scene(scene1)
# r1 = builder.build()

# physx_system.gpu_init()
# physx_system.step()

# ai = physx_system.gpu_create_articulation_index_buffer([r0, r1])
# print("ai", ai)

# qpos_buffer = physx_system.gpu_create_articulation_q_buffer()
# print("qpos", qpos_buffer.shape)

# qvel_buffer = physx_system.gpu_create_articulation_q_buffer()
# print("qvel", qvel_buffer.shape)

# offset_buffer = physx_system.gpu_create_articulation_offset_buffer()
# print("offset", offset_buffer)

# root_pose_buffer = physx_system.gpu_create_articulation_root_pose_buffer()
# print("root_pose", root_pose_buffer.shape)

# physx_system.gpu_query_articulation_qpos(qpos_buffer, ai)
# physx_system.gpu_query_articulation_qvel(qvel_buffer, ai)
# print("qpos", qpos_buffer)
# print("qvel", qvel_buffer)

# physx_system.gpu_query_articulation_root_pose(root_pose_buffer, ai, offset_buffer)
# print("root pose", root_pose_buffer)

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
