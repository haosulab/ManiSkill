from .scene_builder import ReplicaCADRearrangeSceneBuilder


class ReplicaCADTidyHouseTrainSceneBuilder(ReplicaCADRearrangeSceneBuilder):
    task_names = ["tidy_house:train"]


class ReplicaCADTidyHouseValSceneBuilder(ReplicaCADRearrangeSceneBuilder):
    task_names = ["tidy_house:val"]


class ReplicaCADPrepareGroceriesTrainSceneBuilder(ReplicaCADRearrangeSceneBuilder):
    task_names = ["prepare_groceries:train"]


class ReplicaCADPrepareGroceriesValSceneBuilder(ReplicaCADRearrangeSceneBuilder):
    task_names = ["prepare_groceries:val"]


class ReplicaCADSetTableTrainSceneBuilder(ReplicaCADRearrangeSceneBuilder):
    task_names = ["set_table:train"]


class ReplicaCADSetTableValSceneBuilder(ReplicaCADRearrangeSceneBuilder):
    task_names = ["set_table:val"]
