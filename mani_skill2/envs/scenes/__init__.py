from mani_skill2.utils.registration import register_env
from mani_skill2.utils.scene_builder.ai2thor.variants import (
    ArchitecTHORSceneBuilder,
    ProcTHORSceneBuilder,
    RoboTHORSceneBuilder,
    iTHORSceneBuilder,
)
from mani_skill2.utils.scene_builder.replicacad.scene_builder import (
    ReplicaCADSceneBuilder,
)

from .base_env import SceneManipulationEnv

scene_builders = {
    "ReplicaCAD": ReplicaCADSceneBuilder,
    "ArchitecTHOR": ArchitecTHORSceneBuilder,
    "ProcTHOR": ProcTHORSceneBuilder,
    "RoboTHOR": RoboTHORSceneBuilder,
    "iTHOR": iTHORSceneBuilder,
}

# Register environments just for benchmarking/exploration and to be creatable by just ID, these don't have any specific tasks designed in them.
for k, scene_builder in scene_builders.items():
    register_env(
        f"{k}_SceneManipulation-v1",
        max_episode_steps=None,
        scene_builder_cls=scene_builder,
    )(SceneManipulationEnv)
