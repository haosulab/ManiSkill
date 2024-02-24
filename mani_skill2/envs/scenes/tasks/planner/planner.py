from mani_skill2.utils.structs.actor import Actor

from omegaconf import OmegaConf
from dacite import from_dict
from pathlib import Path
from dataclasses import dataclass, field
from typing import Union, Tuple, List
import shortuuid

"""
Task Planner Dataclasses
"""

@dataclass
class Subtask:
    uid: str = field(init=False)
    def __post_init__(self):
        assert self.type in ["pick", "place"]
        self.uid = self.type + '_' + shortuuid.ShortUUID().random(length=6)

TaskPlan = List[Subtask]

@dataclass
class SubtaskConfig:
    task_id: int
    horizon: int = -1
    def __post_init__(self):
        assert self.horizon > 0


@dataclass
class PickSubtask(Subtask):
    obj_id: str
    type: str = "pick"

@dataclass
class PickSubtaskConfig(SubtaskConfig):
    task_id: int = 0
    horizon: int = 200
    ee_rest_thresh: float = 0.05
    def __post_init__(self):
        assert self.ee_rest_thresh >= 0


@dataclass
class PlaceSubtask(Subtask):
    obj_id: str
    goal_pos: Union[str, Tuple[float, float, float], List[Tuple[float, float, float]]]
    type: str = "place"

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.goal_pos, str):
            self.goal_pos = [float(coord) for coord in self.goal_pos.split(',')]

@dataclass
class PlaceSubtaskConfig(SubtaskConfig):
    task_id: int = 1
    horizon: int = 200
    ee_rest_thresh: float = 0.05
    obj_goal_thresh: float = 0.15
    def __post_init__(self):
        assert self.obj_goal_thresh >= 0
        assert self.ee_rest_thresh >= 0

"""
Reading Task Plan from file
"""

@dataclass
class PlanData:
    scene_idx: int
    dataset: str
    plan: TaskPlan

@dataclass
class PlanMetadata:
    scene_idx: int
    dataset: str

def plan_data_from_file(cfg_path: str = None) -> Tuple[TaskPlan, PlanMetadata]:
    cfg_path: Path = Path(cfg_path)
    assert cfg_path.exists(), f"Path {cfg_path} not found"

    plan_data: PlanData = OmegaConf.load(cfg_path)

    plan = []
    for subtask in plan_data.plan:
        if subtask.type == "pick":
            cls = PickSubtask
        elif subtask.type == "place":
            cls = PlaceSubtask
        else:
            raise NotImplementedError(f"Subtask {subtask.type} not implemented yet")
        plan.append(from_dict(data_class=cls, data=subtask))

    metadata = PlanMetadata(scene_idx=plan_data.scene_idx, dataset=plan_data.dataset)

    return plan, metadata

if __name__ == '__main__':
    print(plan_data_from_file(Path(__file__).parent.parent.parent.parent / "task_plans/scene_6_pick_apple.yml"))
