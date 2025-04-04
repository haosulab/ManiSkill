"""
Code to automatically generate robot documentation from the robot classes exported in the mani_skill.agents.robots module.
"""

from typing import List
import urllib.request

import numpy as np
import sapien
from mani_skill.agents.base_agent import BaseAgent
import mani_skill.envs
from mani_skill.envs.tasks.empty_env import EmptyEnv
from mani_skill.utils import sapien_utils
from mani_skill.utils.download_demo import DATASET_SOURCES
from mani_skill.utils.registration import REGISTERED_ENVS
import mani_skill.agents.robots as robots
import gymnasium as gym
import inspect
from pathlib import Path
import cv2

def main():
    # Print all classes exported in the robots module (__init__.py)
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
        try:
            env = EmptyEnv(robot_uids=agent.uid, render_mode="rgb_array", human_render_camera_configs=dict(shader_pack="rt"))
            env.reset()
            kf = env.agent.keyframes
            # Get the first keyframe if available
            if kf and len(kf) > 0:
                first_keyframe_name = next(iter(kf))
                first_keyframe = kf[first_keyframe_name]
                env.agent.robot.set_qpos(first_keyframe.qpos)
                env.agent.robot.set_pose(first_keyframe.pose)

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
            img = env.unwrapped.render_rgb_array().cpu().numpy()[0]
            cv2.imwrite(f"source/robots/images/{agent.uid}_front.png", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            target_pos = (bounds[1][0] + bounds[0][0]) / 2, (bounds[1][1] + bounds[0][1]) / 2, (bounds[1][2] + bounds[0][2]) / 2
            x = largest_side * 1.5 / 2
            pose = sapien_utils.look_at([np.sqrt(1.5) * x, -np.sqrt(1.5) * x, target_pos[2] + np.sqrt(1.5) * x / np.sqrt(3)], target_pos)
            env.scene.human_render_cameras["render_camera"].camera.set_local_pose(pose.sp)
            img = env.unwrapped.render_rgb_array().cpu().numpy()[0]
            cv2.imwrite(f"source/robots/images/{agent.uid}_side.png", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"Error processing {agent.uid}, skipping: {e}")
if __name__ == "__main__":
    main()
