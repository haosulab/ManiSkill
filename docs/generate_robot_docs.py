"""
Code to automatically generate robot documentation from the robot classes exported in the mani_skill.agents.robots module.
"""

from typing import List

import numpy as np
import sapien
import gymnasium as gym

import inspect
from pathlib import Path
import cv2

import mani_skill.envs
import mani_skill.agents.robots as robots
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.tasks.empty_env import EmptyEnv
from mani_skill.utils import sapien_utils
from mani_skill.utils.download_demo import DATASET_SOURCES
import sapien.utils.viewer.entity_window

def capture_images(env: EmptyEnv):
    mesh = env.agent.robot.get_first_visual_mesh()
    bounds = mesh.bounds
    # guess a good camera pose. We want to view the robot from a few angles:
    # from the front (0), 30 around y-axis, 0 around x-axis, 45 around z axis
    # and such that the robot fills up approximately 80% of the image
    # from the front perspective, z-axis is "up", -y is left, +y is right, -x is backward, +x is backward.

    target_pos = (bounds[1] + bounds[0])/2
    largest_side = max(bounds[1] - bounds[0])
    pose = sapien_utils.look_at([largest_side * 1.5, target_pos[1], target_pos[2]], target_pos)
    env.scene.human_render_cameras["render_camera"].camera.set_local_pose(pose.sp)
    img_front = env.unwrapped.render_rgb_array().cpu().numpy()[0]

    target_pos = (bounds[1][0] + bounds[0][0]) / 2, (bounds[1][1] + bounds[0][1]) / 2, (bounds[1][2] + bounds[0][2]) / 2
    x = largest_side * 1.5 / 2
    pose = sapien_utils.look_at([np.sqrt(1.5) * x, -np.sqrt(1.5) * x, target_pos[2] + np.sqrt(1.5) * x / np.sqrt(3)], target_pos)
    env.scene.human_render_cameras["render_camera"].camera.set_local_pose(pose.sp)
    img_side = env.unwrapped.render_rgb_array().cpu().numpy()[0]
    return dict(front=img_front, side=img_side)

def main():
    print("Classes exported in mani_skill.agents.robots:")

    agent_classes: List[BaseAgent] = []
    # Get all attributes in the robots module
    for name, obj in inspect.getmembers(robots):
        # Check if the object is a class
        if inspect.isclass(obj):
            # Check if it's directly from the robots package (not imported from elsewhere)
            if obj.__module__.startswith('mani_skill.agents.robots'):
                print(f"- {name}")
                agent_classes.append(obj)

    for agent in agent_classes:
        if agent.uid != "xarm6_robotiq":
            continue
        try:
            env = EmptyEnv(robot_uids=agent.uid, render_mode="rgb_array", human_render_camera_configs=dict(shader_pack="rt", width=512, height=512))
            env.reset()
            kf = env.agent.keyframes
            # Get the first keyframe if available
            if kf and len(kf) > 0:
                first_keyframe_name = next(iter(kf))
                first_keyframe = kf[first_keyframe_name]
                env.agent.robot.set_qpos(first_keyframe.qpos)
                env.agent.robot.set_pose(first_keyframe.pose)


            imgs = capture_images(env)
            cv2.imwrite(f"source/robots/images/{agent.uid}_front.png", cv2.cvtColor(imgs["front"], cv2.COLOR_BGR2RGB))
            cv2.imwrite(f"source/robots/images/{agent.uid}_side.png", cv2.cvtColor(imgs["side"], cv2.COLOR_BGR2RGB))

            red_mat = sapien.render.RenderMaterial(base_color=[1, 0, 0, 1])
            green_mat = sapien.render.RenderMaterial(base_color=[0, 1, 0, 1])
            blue_mat = sapien.render.RenderMaterial(base_color=[0, 0, 1, 1])
            def add_collision_visual(entity: sapien.Entity):
                new_visual = sapien.render.RenderBodyComponent()
                new_visual.disable_render_id()  # avoid it interfere with visual id counting
                for c in entity.components:
                    if isinstance(c, sapien.physx.PhysxRigidBaseComponent):
                        for s in c.collision_shapes:
                            if isinstance(s, sapien.physx.PhysxCollisionShapeSphere):
                                vs = sapien.render.RenderShapeSphere(s.radius, blue_mat)

                            elif isinstance(s, sapien.physx.PhysxCollisionShapeBox):
                                vs = sapien.render.RenderShapeBox(s.half_size, blue_mat)

                            elif isinstance(s, sapien.physx.PhysxCollisionShapeCapsule):
                                vs = sapien.render.RenderShapeCapsule(
                                    s.radius, s.half_length, blue_mat
                                )

                            elif isinstance(s, sapien.physx.PhysxCollisionShapeConvexMesh):
                                vs = sapien.render.RenderShapeTriangleMesh(
                                    s.vertices,
                                    s.triangles,
                                    np.zeros((0, 3)),
                                    np.zeros((0, 2)),
                                    green_mat,
                                )
                                vs.scale = s.scale

                            elif isinstance(s, sapien.physx.PhysxCollisionShapeTriangleMesh):
                                vs = sapien.render.RenderShapeTriangleMesh(
                                    s.vertices,
                                    s.triangles,
                                    np.zeros((0, 3)),
                                    np.zeros((0, 2)),
                                    red_mat,
                                )
                                vs.scale = s.scale

                            elif isinstance(s, sapien.physx.PhysxCollisionShapePlane):
                                vs = sapien.render.RenderShapePlane([1, 1e4, 1e4], blue_mat)

                            elif isinstance(s, sapien.physx.PhysxCollisionShapeCylinder):
                                vs = sapien.render.RenderShapeCylinder(
                                    s.radius, s.half_length, green_mat
                                )

                            else:
                                raise Exception(
                                    "invalid collision shape, this code should be unreachable."
                                )

                            vs.local_pose = s.local_pose

                            new_visual.attach(vs)

                entity.add_component(new_visual)
                new_visual.set_property("shadeFlat", 1)
            for link in env.agent.robot.links:
                for c in link._objs[0].entity.components:
                    if isinstance(c, sapien.render.RenderBodyComponent):
                        c.disable()
                add_collision_visual(link._objs[0].entity)
            imgs = capture_images(env)
            cv2.imwrite(f"source/robots/images/{agent.uid}_front_collision.png", cv2.cvtColor(imgs["front"], cv2.COLOR_BGR2RGB))
            cv2.imwrite(f"source/robots/images/{agent.uid}_side_collision.png", cv2.cvtColor(imgs["side"], cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"Error processing {agent.uid}, skipping: {e}")
if __name__ == "__main__":
    main()
