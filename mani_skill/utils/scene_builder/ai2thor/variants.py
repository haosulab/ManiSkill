from mani_skill.utils.scene_builder.registration import register_scene_builder

from .scene_builder import AI2THORBaseSceneBuilder


class ProcTHORSceneBuilder(AI2THORBaseSceneBuilder):
    scene_dataset = "ProcTHOR"


@register_scene_builder("ArchitecTHOR")
class ArchitecTHORSceneBuilder(AI2THORBaseSceneBuilder):
    scene_dataset = "ArchitecTHOR"


class iTHORSceneBuilder(AI2THORBaseSceneBuilder):
    scene_dataset = "iTHOR"


class RoboTHORSceneBuilder(AI2THORBaseSceneBuilder):
    scene_dataset = "RoboTHOR"
