import numpy as np
import sapien

from mani_skill.utils.building._mjcf_loader import MJCFLoader

scene = sapien.Scene()
scene.set_ambient_light([0.3, 0.3, 0.3])
scene.add_directional_light([0, -1, -1], [0.85, 0.85, 0.85], True)
# scene.add_ground(0)
# scene.render_system.set_cubemap(sapien.render.RenderCubemap())

loader = MJCFLoader()
loader.set_scene(scene)
loader.fix_root_link = True
robot = loader.load("manualtest/assets/mujoco/cartpole.xml")
robot.joints[1].set_drive_properties(stiffness=200, damping=20)
# robot.set_pose(sapien.Pose(p=[0, 0, 1.28]))

viewer = scene.create_viewer()
viewer.set_scene(scene)
viewer.paused = True
while True:
    scene.step()
    robot.joints[1].set_drive_target(1)
    scene.update_render()
    viewer.render()
