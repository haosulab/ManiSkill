# in this file, we compute the drive for the robotiq 140. 
# The target_p [[0], [-0.07921], [-0.13704]] is computed by setting point in the Omniverse Platform to get more accurate estimation.
# An intuitive method is also implemented in this file to roughly obtain approximate values for p_p and p_f.

from __future__ import annotations

import os
import sapien
from mani_skill.envs.utils.system.backend import parse_sim_and_render_backend

from mani_skill.envs.scene import ManiSkillScene
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.pose import Pose
import numpy as np

def get_robot_link_constraint(robot: sapien.Articulation, scene: sapien.Scene):
    right_outer_knuckle_joint = next(
        j for j in robot.get_active_joints() if j.name == "right_outer_knuckle_joint"
    )
    right_inner_finger_joint = next(
        j for j in robot.get_active_joints() if j.name == "right_inner_finger_joint"
    )
    right_inner_knuckle_joint = next(
        j for j in robot.get_active_joints() if j.name == "right_inner_knuckle_joint"
    )
    right_inner_finger_pad_joint = next(
        j for j in robot.get_joints() if j.name == "right_inner_finger_pad_joint"
    )


    # print("yes")
    print(f"{right_outer_knuckle_joint.name}.get_global_pose().p", right_outer_knuckle_joint.get_global_pose().p)
    print(f"{right_inner_finger_joint.name}.get_global_pose().p", right_inner_finger_joint.get_global_pose().p)
    print(f"{right_inner_knuckle_joint.name}.get_global_pose().p", right_inner_knuckle_joint.get_global_pose().p)
    print(f"{right_inner_finger_pad_joint.name}.get_global_pose().p", right_inner_finger_pad_joint.get_global_pose().p)

    # target_p = (right_inner_finger_joint.get_global_pose().p).numpy().reshape(3, 1)
    target_p = (right_inner_finger_joint.get_global_pose().p).numpy().reshape(3, 1)
    target_p[1] = -right_inner_finger_pad_joint.get_global_pose().p[0][1]

    # target_p = np.array([[
        # 0], [-0.07921], [-0.13704]
    # ])
    print("target_p", target_p)

    right_inner_knuckle = right_inner_knuckle_joint.get_child_link()
    right_inner_finger = right_inner_finger_joint.get_child_link()

    right_inner_knuckle_pose = right_inner_knuckle_joint.get_global_pose()
    right_inner_finger_pose = right_inner_finger_joint.get_global_pose()
    
    T_pw = right_inner_finger_pose.inv().to_transformation_matrix()[0]
    # import pdb; pdb.set_trace()
    print("T_pw", T_pw)
    # Ensure target_p is a 1D array of shape (3,) for correct matrix multiplication
    # print(T_pw[:3, :3].shape)
    target_p_vec = target_p
    print("T_pw", T_pw[:3, :3])
    we_need_joint_in_right_inner_finger = T_pw[:3, :3] @ target_p_vec + T_pw[:3, [3]]

    T_fw = right_inner_knuckle_pose.inv().to_transformation_matrix()[0]
    print("T_fw", T_fw)
    we_need_joint_in_right_inner_knuckle = T_fw[:3, :3] @ target_p_vec + T_fw[:3, [3]]
    
    
    we_need_joint_in_right_inner_knuckle = we_need_joint_in_right_inner_knuckle.flatten()
    we_need_joint_in_right_inner_finger = we_need_joint_in_right_inner_finger.flatten()
    scene.create_drive(right_inner_knuckle, sapien.Pose(we_need_joint_in_right_inner_knuckle), right_inner_finger, sapien.Pose(we_need_joint_in_right_inner_finger))

    print("we_need_joint_in_right_inner_knuckle", we_need_joint_in_right_inner_knuckle)
    print("we_need_joint_in_right_inner_finger", we_need_joint_in_right_inner_finger)
    
    
    left_inner_finger_joint = next(
        j for j in robot.get_active_joints() if j.name == "left_inner_finger_joint"
    )
    left_inner_knuckle_joint = next(
        j for j in robot.get_active_joints() if j.name == "left_inner_knuckle_joint"
    )
    left_inner_finger_pad_joint = next(
        j for j in robot.get_joints() if j.name == "left_inner_finger_pad_joint"
    )
    
    left_inner_knuckle = left_inner_knuckle_joint.get_child_link()
    left_inner_finger = left_inner_finger_joint.get_child_link()
    
    left_inner_knuckle_pose = left_inner_knuckle_joint.get_global_pose()
    left_inner_finger_pose = left_inner_finger_joint.get_global_pose()
    
    T_pw = left_inner_finger_pose.inv().to_transformation_matrix()[0]
    print("T_pw", T_pw)
    
    # target_p_left_vec = np.array([[
    #     0], [0.07921], [-0.13704]
    # ])
    target_p_left_vec = (left_inner_finger_joint.get_global_pose().p).numpy().reshape(3, 1)
    target_p_left_vec[1] = -left_inner_finger_pad_joint.get_global_pose().p[0][1]
    
    T_fw = left_inner_knuckle_pose.inv().to_transformation_matrix()[0]
    print("T_fw", T_fw)
    we_need_joint_in_left_inner_knuckle = T_fw[:3, :3] @ target_p_left_vec + T_fw[:3, [3]]
    
    
    we_need_joint_in_left_inner_finger = T_pw[:3, :3] @ target_p_left_vec + T_pw[:3, [3]]


    we_need_joint_in_left_inner_knuckle = we_need_joint_in_left_inner_knuckle.flatten()
    we_need_joint_in_left_inner_finger = we_need_joint_in_left_inner_finger.flatten()
    
    print("we_need_joint_in_left_inner_knuckle", we_need_joint_in_left_inner_knuckle)
    print("we_need_joint_in_left_inner_finger", we_need_joint_in_left_inner_finger)
    
    
    scene.create_drive(left_inner_knuckle, sapien.Pose(we_need_joint_in_left_inner_knuckle), left_inner_finger, sapien.Pose(we_need_joint_in_left_inner_finger))


def load_xarm_robotiq_urdf(
    urdf_path: str = "assets/robots/xarm6_robotiq_140/xarm_robotiq_arg2f_140_model.urdf",
    enable_viewer: bool = True,
):
    """Load the xArm + Robotiq 140 URDF into a ManiSkill scene and return the scene and articulation.

    Args:
        urdf_path: Absolute path to the `.urdf` file.

    Returns:
        (scene, robot): The ManiSkill scene and the loaded articulation wrapper.
    """
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.abspath(urdf_path)
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    # Create a ManiSkill scene with explicit CPU sim & CPU render backend
    backend = parse_sim_and_render_backend(sim_backend="cpu", render_backend="sapien_cpu")
    scene = ManiSkillScene(backend=backend)
    scene.set_timestep(1.0 / 240.0)
    
    sapien.physx.set_scene_config(gravity=[0,0,0])
    
    # Load URDF
    loader = scene.create_urdf_loader()
    loader.name = "xarm_robotiq_140"
    loader.fix_root_link = True
    loader.disable_self_collisions = False
    robot = loader.load(urdf_path, name=loader.name)

    for link in robot.get_links():
        link.disable_gravity = True
        
    # Basic lighting and ground for visualization
    scene.set_ambient_light([0.3, 0.3, 0.3])
    scene.add_directional_light([1, 1, -1], [1, 1, 1], shadow=True, shadow_scale=5)
    scene.sub_scenes[0].add_ground(-0.9, render=True)

    # Place the robot at the origin (adjust if needed)
    robot.set_pose(sapien.Pose([0.0, 0.0, 0.0]))
    get_robot_link_constraint(robot, scene)
    human_cam_eye_pos = [0.6,-0.7,-0.6]  # human cam is the camera used for human rendering (i.e. eval videos)
    human_cam_target_pos = [0.0, 0.0, 0.35]
    # Optional viewer
    viewer = None
    if enable_viewer and scene.can_render():

        pose = sapien_utils.look_at(
            eye=human_cam_eye_pos, target=human_cam_target_pos
        )
        viewer_cfg =CameraConfig("render_camera", pose, 1024, 1024, 1, 0.01, 100)
    
        viewer = sapien_utils.create_viewer(viewer_cfg)
        viewer.set_scene(scene.sub_scenes[0])

    return scene, robot, viewer


def main():
    scene, robot, viewer = load_xarm_robotiq_urdf()
    print(f"Loaded articulation: {robot.name}")
    print(f"Num links: {len(robot.get_links())}, Num joints: {len(robot.get_active_joints())}")

    # Interactive render loop if viewer is available
    if viewer is not None:
        while not viewer.closed:
            scene.update_render()
            viewer.render()
            scene.step()
    else:
        # Headless fallback: just step and update render
        for _ in range(10000):
            scene.step()
            scene.update_render()


if __name__ == "__main__":
    main()


