import numpy as np
import sapien

from mani_skill.utils.building.mjcf_loader import MJCFLoader

scene = sapien.Scene()
scene.set_ambient_light([0.3, 0.3, 0.3])
scene.add_directional_light([0, 0, -1], [0.2, 0.2, 0.2], True)
scene.add_ground(0)


loader = MJCFLoader()
loader.set_scene(scene)
loader.load("humanoid.xml")


viewer = scene.create_viewer()
viewer.set_scene(scene)
while True:
    scene.step()
    scene.update_render()
    viewer.render()
