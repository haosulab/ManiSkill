import sapien.core as sapien
from mani_skill2 import ASSET_DIR


def minimalistic_modern_bedroom(scene):
    builder = scene.create_actor_builder()
    # "Minimalistic Modern Bedroom" (https://skfb.ly/oCnNx) by dylanheyes is licensed under Creative Commons Attribution (http://creativecommons.org/licenses/by/4.0/).
    path = ASSET_DIR / "background/minimalistic_modern_bedroom.glb"
    if not path.exists():
        print("Please download the background assets by `python -m mani_skill2.utils.download_background`")
        exit(1)
    builder.add_visual_from_file(str(path))
    arena = builder.build_kinematic()
    arena.set_pose(sapien.Pose([0, 0, 1.7], [0.5, 0.5, -0.5, -0.5]))
    scene.set_ambient_light([0.1, 0.1, 0.1])
    scene.add_point_light([-0.349, 0, 1.4], [1.0, 0.9, 0.9])
    return arena
