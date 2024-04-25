from .scene_builder import AI2THORBaseSceneBuilder


class ProcTHORSceneBuilder(AI2THORBaseSceneBuilder):
    scene_dataset = "ProcTHOR"


class ArchitecTHORSceneBuilder(AI2THORBaseSceneBuilder):
    scene_dataset = "ArchitecTHOR"


class iTHORSceneBuilder(AI2THORBaseSceneBuilder):
    scene_dataset = "iTHOR"


class RoboTHORSceneBuilder(AI2THORBaseSceneBuilder):
    scene_dataset = "RoboTHOR"
