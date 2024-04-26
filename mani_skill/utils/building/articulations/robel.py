from typing import Sequence

import numpy as np
import sapien
import sapien.physx as physx

from mani_skill.envs.scene import ManiSkillScene


def build_robel_valve(
    scene: ManiSkillScene,
    valve_angles: Sequence[float],
    name: str,
    radius_scale: float = 1.0,
    capsule_radius_scale: float = 1.0,
    scene_idxs=None,
):
    # Size and geometry of valve are based on the original setting of Robel benchmark, unit: m
    # Ref: https://github.com/google-research/robel
    capsule_height = 0.039854
    capsule_length = 0.061706 * radius_scale
    capsule_radius = 0.0195 * capsule_radius_scale
    bottom_length = 0.04
    bottom_height = 0.03
    bearing_radius = 0.007
    bearing_height = 0.032

    builder = scene.create_articulation_builder()
    builder.set_scene_idxs(scene_idxs)

    # Mount link
    mount_builder = builder.create_link_builder(parent=None)
    mount_builder.set_name("mount")
    mount_builder.add_box_collision(
        pose=sapien.Pose([0, 0, bottom_height / 2]),
        half_size=[bottom_length / 2, bottom_length / 2, bottom_height / 2],
    )
    mount_builder.add_box_visual(
        pose=sapien.Pose([0, 0, bottom_height / 2]),
        half_size=[bottom_length / 2, bottom_length / 2, bottom_height / 2],
    )
    mount_builder.add_cylinder_visual(
        pose=sapien.Pose(
            [0, 0, bottom_height + bearing_height / 2], [-0.707, 0, 0.707, 0]
        ),
        half_length=bottom_height / 2,
        radius=bearing_radius,
    )
    mount_builder.add_cylinder_collision(
        pose=sapien.Pose(
            [0, 0, bottom_height + bearing_height / 2], [-0.707, 0, 0.707, 0]
        ),
        half_length=bottom_height / 2,
        radius=bearing_radius,
    )

    # Valve link
    valve_builder = builder.create_link_builder(mount_builder)
    valve_builder.set_name("valve")
    valve_angles = np.array(valve_angles)
    if np.min(valve_angles) < 0 or np.max(valve_angles) > 2 * np.pi:
        raise ValueError(
            f"valve_angles should be within 0-2*pi, but got {valve_angles}"
        )

    for i, angle in enumerate(valve_angles):
        rotate_pose = sapien.Pose([0, 0, 0])
        rotate_pose.set_rpy([0, 0, angle])
        capsule_pose = rotate_pose * sapien.Pose([capsule_length / 2, 0, 0])
        color = np.array([1, 1, 1, 1]) if i > 0 else np.array([1, 0, 0, 1])
        viz_mat = sapien.render.RenderMaterial(
            base_color=color, roughness=0.5, specular=0.5
        )
        valve_builder.add_capsule_visual(
            pose=capsule_pose,
            radius=capsule_radius,
            half_length=capsule_length / 2,
            material=viz_mat,
        )
        physx_mat = physx.PhysxMaterial(1, 0.8, 0)
        valve_builder.add_capsule_collision(
            pose=capsule_pose,
            radius=capsule_radius,
            half_length=capsule_length / 2,
            material=physx_mat,
            patch_radius=0.1,
            min_patch_radius=0.03,
        )

    valve_builder.set_joint_name("valve_joint")
    valve_builder.set_joint_properties(
        type="revolute",
        limits=[[-np.inf, np.inf]],
        pose_in_parent=sapien.Pose(
            [0, 0, capsule_height + bottom_height], [0.707, 0, 0.707, 0]
        ),
        pose_in_child=sapien.Pose(q=[0.707, 0, 0.707, 0]),
        friction=0.02,
        damping=2,
    )

    valve = builder.build(name, fix_root_link=True)
    return valve, capsule_length
