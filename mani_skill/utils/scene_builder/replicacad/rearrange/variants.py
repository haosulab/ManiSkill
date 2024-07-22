from .scene_builder import ReplicaCADRearrangeSceneBuilder
from mani_skill.utils.scene_builder.registration import register_scene_builder


@register_scene_builder("ReplicaCADTidyHouseTrain")
class ReplicaCADTidyHouseTrainSceneBuilder(ReplicaCADRearrangeSceneBuilder):
    task_names = ["tidy_house:train"]


@register_scene_builder("ReplicaCADTidyHouseVal")
class ReplicaCADTidyHouseValSceneBuilder(ReplicaCADRearrangeSceneBuilder):
    task_names = ["tidy_house:val"]


@register_scene_builder("ReplicaCADPrepareGroceriesTrain")
class ReplicaCADPrepareGroceriesTrainSceneBuilder(ReplicaCADRearrangeSceneBuilder):
    task_names = ["prepare_groceries:train"]


@register_scene_builder("ReplicaCADPrepareGroceriesVal")
class ReplicaCADPrepareGroceriesValSceneBuilder(ReplicaCADRearrangeSceneBuilder):
    task_names = ["prepare_groceries:val"]


@register_scene_builder("ReplicaCADSetTableTrain")
class ReplicaCADSetTableTrainSceneBuilder(ReplicaCADRearrangeSceneBuilder):
    task_names = ["set_table:train"]


@register_scene_builder("ReplicaCADSetTableVal")
class ReplicaCADSetTableValSceneBuilder(ReplicaCADRearrangeSceneBuilder):
    task_names = ["set_table:val"]
