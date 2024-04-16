import numpy as np
import sapien

from mani_skill.utils.building.mjcf_loader import MJCFLoader

scene = sapien.Scene()
scene.set_ambient_light([0.3, 0.3, 0.3])
scene.add_directional_light([0, -1, -1], [0.85, 0.85, 0.85], True)
scene.add_ground(0)


loader = MJCFLoader()
loader.set_scene(scene)
robot = loader.load("cheetah.xml")

# robot.set_pose(sapien.Pose(p=[0, 0, 1.28]))

viewer = scene.create_viewer()
viewer.set_scene(scene)
viewer.paused = True
while True:
    scene.step()
    scene.update_render()
    viewer.render()
