import json
import os.path as osp
from dataclasses import dataclass
from pathlib import Path

from mani_skill import ASSET_DIR


@dataclass
class AI2BuildConfig:
    config_file: str
    source: str
    spawn_pos_file: str = None


@dataclass
class SceneDataset:
    metadata_path: str
    dataset_path: str


def load_ai2thor_metadata():
    with open(
        str(
            Path(ASSET_DIR)
            / "scene_datasets/ai2thor/ai2thor-hab/configs/object_semantic_id_mapping.json"
        ),
        "r",
    ) as f:
        OBJECT_SEMANTIC_ID_MAPPING = json.load(f)
        SEMANTIC_ID_OBJECT_MAPPING = dict()
        for k in OBJECT_SEMANTIC_ID_MAPPING:
            SEMANTIC_ID_OBJECT_MAPPING[OBJECT_SEMANTIC_ID_MAPPING[k]] = k
    MOVEABLE_OBJECT_IDS = [
        "Apple",
        "AppleSliced",
        "Tomato",
        "TomatoSliced",
        "Bread",
        "BreadSliced",
        "Chair",
        "HousePlant",
        "Pot",
        "Pan",
        "Knife",
        "Fork",
        "Spoon",
        "Bowl",
        "Toaster",
        "CoffeeMachine",
        "Egg",
        "Lettuce",
        "Potato",
        "Mug",
        "Plate",
        "GarbageCan",
        "Omelette",
        "EggShell",
        "EggCracked",
        "Container",
        "Cup",
        "ButterKnife",
        "PotatoSliced",
        "MugFilled",
        "BowlFilled",
        "LettuceSliced",
        "ContainerFull",
        "BowlDirty",
        "Sandwich",
        "TissueBox",
        "VacuumCleaner",
        "WateringCan",
        "Laptop",
        "RemoteControl",
        "Box",
        "Newspaper",
        "KeyChain",
        "Dirt",
        "CellPhone",
        "CreditCard",
        "Cloth",
        "Candle",
        "Plunger",
        "ToiletPaper",
        "ToiletPaperHanger",
        "SoapBottle",
        "SoapBottleFilled",
        "SoapBar",
        "ShowerDoor",
        "SprayBottle",
        "ScrubBrush",
        "ToiletPaperRoll",
        "Lamp",
        "Book",
        "SportsEquipment",
        "Pen",
        "Pencil",
        "Watch",
        "MiscTableObject",
        "BaseballBat",
        "BasketBall",
        "Boots",
        "Bottle",
        "DishSponge",
        "FloorLamp",
        "Kettle",
        "Lighter",
        "PanLid",
        "PaperTowelRoll",
        "PepperShaker",
        "PotLid",
        "SaltShaker",
        "Safe",
        "SmallMirror",
        "Spatula",
        "TeddyBear",
        "TennisRacket",
        "Tissue",
        "Vase",
        "MassObjectSpawner",
        "MassScale",
        "Footstool",
        "Pillow",
        "Cart",
        "DeskLamp",
        "CD",
        "Poster",
        "HandTowel",
        "Ladle",
        "WineBottle",
        "AluminumFoil",
        "DogBed",
        "Dumbbell",
        "TableTopDecor",
        "RoomDecor",
        "Stool",
        "GarbageBag",
        "Desktop",
        "TargetCircle",
    ]
    return OBJECT_SEMANTIC_ID_MAPPING, SEMANTIC_ID_OBJECT_MAPPING, MOVEABLE_OBJECT_IDS


# This maps a scene set e.g. ProcTHOR to an adapter, metadata, and where the scenes are saved to. The adapter is a class that can load the scene set
SCENE_SOURCE_TO_DATASET: dict[str, SceneDataset] = {
    "ProcTHOR": SceneDataset(
        metadata_path="ProcTHOR.json",
        dataset_path=osp.join(
            ASSET_DIR, "scene_datasets/ai2thor/ai2thor-hab/configs/scenes/ProcTHOR"
        ),
    ),
    "ArchitecTHOR": SceneDataset(
        metadata_path="ArchitecTHOR.json",
        dataset_path=osp.join(
            ASSET_DIR, "scene_datasets/ai2thor/ai2thor-hab/configs/scenes/ArchitecTHOR"
        ),
    ),
    "iTHOR": SceneDataset(
        metadata_path="iTHOR.json",
        dataset_path=osp.join(
            ASSET_DIR, "scene_datasets/ai2thor/ai2thor-hab/configs/scenes/iTHOR"
        ),
    ),
    "RoboTHOR": SceneDataset(
        metadata_path="RoboTHOR.json",
        dataset_path=osp.join(
            ASSET_DIR, "scene_datasets/ai2thor/ai2thor-hab/configs/scenes/RoboTHOR"
        ),
    ),
}
