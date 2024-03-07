from dataclasses import dataclass
from typing import Dict

from mani_skill import logger
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.utils.scene_builder import scene_builder


@dataclass
class SceneBuilderSpec:
    """Scene builder specifications. At the moment it is a simple wrapper around the scene_builder_cls but the dataclass is used in case we may need additional metadata"""

    scene_builder_cls: type[scene_builder.SceneBuilder]


REGISTERED_SCENE_BUILDERS: Dict[str, SceneBuilderSpec] = {}


def register_scene_builder(uid: str, override=False):
    """A decorator to register scene builders into ManiSkill so they can be used easily by string uid.

    Args:
        uid (str): unique id of the scene builder class.
        override (bool): whether to override the scene builder if it is already registered.
    """

    def _register_scene_builder(scene_builder_cls: type[scene_builder.SceneBuilder]):
        if uid in REGISTERED_SCENE_BUILDERS:
            if override:
                logger.warn(f"Overriding registered scene builder {uid}")
                REGISTERED_SCENE_BUILDERS.pop(uid)
            else:
                logger.warn(
                    f"Scene Builder {uid} is already registered. Skip registration."
                )
            return scene_builder_cls

        REGISTERED_SCENE_BUILDERS[uid] = SceneBuilderSpec(
            scene_builder_cls=scene_builder_cls
        )
        return scene_builder_cls

    return _register_scene_builder
