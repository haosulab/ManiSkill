from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.ai2thor.variants import (
    ArchitecTHORSceneBuilder,
    ProcTHORSceneBuilder,
    RoboTHORSceneBuilder,
    iTHORSceneBuilder,
)
from mani_skill.utils.scene_builder.registration import REGISTERED_SCENE_BUILDERS
from mani_skill.utils.scene_builder.replicacad.scene_builder import (
    ReplicaCADSceneBuilder,
)

from .base_env import SceneManipulationEnv

# Register environments just for benchmarking/exploration and to be creatable by just ID, these don't have any specific tasks designed in them.
for k, scene_builder_spec in REGISTERED_SCENE_BUILDERS.items():
    register_env(
        f"{k}_SceneManipulation-v1",
        max_episode_steps=200,
        scene_builder_cls=k,
    )(SceneManipulationEnv)
