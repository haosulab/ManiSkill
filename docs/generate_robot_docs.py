"""
Code to automatically generate robot documentation from the robot classes exported in the mani_skill.agents.robots module.

In the docs/ folder run generate_robot_docs.py to update the robot documentation. If a new robot is added make sure to add a entry to the
metadata/robots.json file which adds details about the robot not really needed in the code like the modelling quality.
"""


GLOBAL_ROBOT_DOCS_HEADER = """<!-- THIS IS ALL GENERATED DOCUMENTATION via generate_robot_docs.py. DO NOT MODIFY THIS FILE DIRECTLY. -->
"""

QUALITY_KEY_TO_DESCRIPTION = {
    "A+": "Values are the product of proper system identification",
    "A": "Values are realistic, but have not been properly identified",
    "B": "Stable, but some values are unrealistic",
    "C": "Conditionally stable, can be significantly improved",
}
import json
from typing import List
import sys

import numpy as np

import inspect
from pathlib import Path
import cv2

import mani_skill.envs
import mani_skill.agents.robots as robots
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.tasks.empty_env import EmptyEnv
from mani_skill.utils import sapien_utils
from mani_skill.utils.download_asset import DATA_GROUPS
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

def main(robot_id: str = None):
    base_dir = Path(__file__).parent / "source/robots"
    robot_metadata = json.load(open(Path(__file__).parent / "metadata/robot.json"))

    agent_classes: List[BaseAgent] = []
    # Get all attributes in the robots module
    for name, obj in inspect.getmembers(robots):
        # Check if the object is a class
        if inspect.isclass(obj):
            # Check if it's directly from the robots package (not imported from elsewhere)
            if obj.__module__.startswith('mani_skill.agents.robots'):
                agent_classes.append(obj)
    robot_index_markdown_str = GLOBAL_ROBOT_DOCS_HEADER + """
# Robots
<img src="../_static/robot_images/robot-grid.png" alt="Robot Grid" style="width: 100%; height: auto;">


This sections here show the already built/modelled robots ready for simulation across a number of different categories. Some of them are displayed above in an empty environment using a predefined keyframe. Note that not all of these robots are used in tasks in ManiSkill, and some are not tuned for maximum efficiency yet or for sim2real transfer. You can generally assume robots that are used in existing tasks in ManiSkill are of the highest quality and already tuned.

To learn about how to load your own custom robots see [the tutorial](../user_guide/tutorials/custom_robots.md).

## Robots Table
Table of all robots modelled in ManiSkill. Click the robot's picture to see more details on the robot, including more views, collision models, controllers implemented and more.

A quality rating is also given for each robot which rates the robot on how well modelled it is. It follows the same scale as [Mujoco Menagerie](https://github.com/google-deepmind/mujoco_menagerie?tab=readme-ov-file#model-quality-and-contributing)

| Grade | Description                                                 |
|-------|-------------------------------------------------------------|
| A+    | Values are the product of proper system identification      |
| A     | Values are realistic, but have not been properly identified |
| B     | Stable, but some values are unrealistic                     |
| C     | Conditionally stable, can be significantly improved         |

Robots that are cannot be stably simulated are not included in ManiSkill at all. Most robots will have a grade of B (essentially does it look normal in simulation). While some robots may have grades of A/A+ we still strongly recommend you perform your own system ID as each robot might be a bit different.

<div class="gallery" style="display: flex; flex-wrap: wrap; gap: 10px;">
"""
    for row_idx, agent in enumerate(agent_classes):
        ## generate images of the robot ###
        print(f"Generating docs for {agent.uid}")

        ### generate robot docs ###
        metadata = robot_metadata.pop(agent.uid, {})
        quality = metadata.get("quality", None)

        requires_downloading_assets = False
        if agent.uid in DATA_GROUPS:
            requires_downloading_assets = True
        urdf_path = agent.urdf_path
        agent_class_code_link = f"https://github.com/haosulab/ManiSkill/blob/main/{agent.__module__.replace('.', '/') + '.py'}"
        if requires_downloading_assets:
            # note (stao): asset download location might move away from github to another place in the future
            pass
        agent_name = metadata.get("name", agent.uid)
        robot_index_markdown_str += (f"""
<div class="gallery-item">
    <a href="{agent.uid}">
        <img src="../_static/robot_images/{agent.uid}/thumbnail.png" style='min-width:min(50%, 100px);max-width:200px;height:auto' alt="{agent_name}">
    </a>
    <div class="gallery-item-caption">
        <a href="{agent.uid}"><p style="margin-bottom: 0px; word-wrap: break-word; max-width: 200px; color: inherit;">{agent_name}</p></a>
        <p style="margin-top: 0px;">Quality: {quality if quality is not None else "N/A"}</p>
    </div>
</div>
"""
        )

        if robot_id is None or robot_id == agent.uid:
            env = EmptyEnv(robot_uids=agent.uid, render_mode="rgb_array", human_render_camera_configs=dict(shader_pack="rt", width=1024, height=1024))
            env.reset()
            robot_dof = env.agent.robot.dof.item()
            controllers = list(env.agent._controller_configs.keys())

            kf = env.agent.keyframes
            # Get the first keyframe if available
            if kf and len(kf) > 0:
                first_keyframe_name = next(iter(kf))
                first_keyframe = kf[first_keyframe_name]
                env.agent.robot.set_qpos(first_keyframe.qpos)
                env.agent.robot.set_pose(first_keyframe.pose)


            imgs = capture_images(env)
            Path(f"source/_static/robot_images/{agent.uid}").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(f"source/_static/robot_images/{agent.uid}/front_visual.png", cv2.cvtColor(imgs["front"], cv2.COLOR_BGR2RGB))
            cv2.imwrite(f"source/_static/robot_images/{agent.uid}/side_visual.png", cv2.cvtColor(imgs["side"], cv2.COLOR_BGR2RGB))

            cv2.imwrite(f"source/_static/robot_images/{agent.uid}/thumbnail.png", cv2.cvtColor(cv2.resize(imgs["side"], (256, 256)), cv2.COLOR_BGR2RGB))

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
            cv2.imwrite(f"source/_static/robot_images/{agent.uid}/front_collision.png", cv2.cvtColor(imgs["front"], cv2.COLOR_BGR2RGB))
            cv2.imwrite(f"source/_static/robot_images/{agent.uid}/side_collision.png", cv2.cvtColor(imgs["side"], cv2.COLOR_BGR2RGB))



            # generate robot specific documentation

            if quality is not None:
                quality_desc = f"{quality} ({QUALITY_KEY_TO_DESCRIPTION[quality]})"
            else:
                quality_desc = "N/A"
            robot_page_markdown_str = GLOBAL_ROBOT_DOCS_HEADER + f"""
# {agent_name}

Robot UID: `{agent.uid}`

Agent Class Code: [{agent_class_code_link}]({agent_class_code_link})

Quality: {quality_desc}

Degrees of Freedom: {robot_dof}

Controllers: {", ".join([f"`{c}`" for c in controllers])}

## Visuals and Collision Models

<div>
    <div style="max-width: 100%; display: flex; justify-content: center;">
        <img src="../../_static/robot_images/{agent.uid}/front_visual.png" style='min-width:min(50%, 100px);max-width:50%;height:auto' alt="{agent.uid}">
        <img src="../../_static/robot_images/{agent.uid}/side_visual.png" style='min-width:min(50%, 100px);max-width:50%;height:auto' alt="{agent.uid}">
    </div>
    <p style="text-align: center; font-size: 1.2rem;">Visual Meshes</p>
    <br/>
    <div style="max-width: 100%; display: flex; justify-content: center;">
        <img src="../../_static/robot_images/{agent.uid}/front_collision.png" style='min-width:min(50%, 100px);max-width:50%;height:auto' alt="{agent.uid}">
        <img src="../../_static/robot_images/{agent.uid}/side_collision.png" style='min-width:min(50%, 100px);max-width:50%;height:auto' alt="{agent.uid}">
    </div>
    <p style="text-align: center; font-size: 1.2rem;">Collision Meshes (Green = Convex Mesh, Blue = Primitive Shape Mesh)</p>
</div>
"""
            Path(f"{base_dir}/{agent.uid}").mkdir(parents=True, exist_ok=True)
            with open(
                f"{base_dir}/{agent.uid}/index.md", "w"
            ) as f:
                f.write(robot_page_markdown_str)
    robot_index_markdown_str += """\n</div>

```{toctree}
:caption: Directory
:maxdepth: 1

"""
    for agent in agent_classes:
        robot_index_markdown_str += f"{agent.uid}/index\n"
    robot_index_markdown_str += """
```"""

    with open(
        f"{base_dir}/index.md", "w"
    ) as f:
        f.write(robot_index_markdown_str)


    if len(robot_metadata) > 0:
        print(f"Warning: {len(robot_metadata)} robots found in the metadata/robot.json file but do not have corresponding robot classes in the mani_skill.agents.robots module")
        for robot in robot_metadata:
            print(f"- {robot}")
if __name__ == "__main__":
    robot_id = sys.argv[1] if len(sys.argv) > 1 else None
    main(robot_id)
