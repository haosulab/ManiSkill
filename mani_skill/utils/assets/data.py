"""
Asset sources and tooling for managing the assets
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from mani_skill import ASSET_DIR, PACKAGE_ASSET_DIR
from mani_skill.utils import io_utils


@dataclass
class DataSource:
    source_type: str
    """what kind of data is this"""
    url: Optional[str] = None
    hf_repo_id: Optional[str] = None
    github_url: Optional[str] = None
    target_path: Optional[str] = None
    """the folder where the file will be downloaded to"""
    checksum: Optional[str] = None
    zip_dirname: Optional[str] = None
    """what to rename a zip files generated directory to"""
    filename: Optional[str] = None
    """name to change the downloaded file to. If None, will not change the name"""
    output_dir: str = ASSET_DIR


DATA_SOURCES: Dict[str, DataSource] = {}
"""Data sources map data source IDs to their respective DataSource objects which contain info on what the data is and where to download it"""
DATA_GROUPS: Dict[str, List[str]] = {}
"""Data groups map group ids (typically environment IDs) to a list of data source/group IDs for easy group management. data groups can be done hierarchicaly"""


def is_data_source_downloaded(data_source_id: str):
    data_source = DATA_SOURCES[data_source_id]
    return os.path.exists(data_source.output_dir / data_source.target_path)


def initialize_data_sources():
    DATA_SOURCES["ycb"] = DataSource(
        source_type="task_assets",
        url="https://huggingface.co/datasets/haosulab/ManiSkill2/resolve/main/data/mani_skill2_ycb.zip",
        target_path="assets/mani_skill2_ycb",
        checksum="174001ba1003cc0c5adda6453f4433f55ec7e804f0f0da22d015d525d02262fb",
    )
    DATA_SOURCES["pick_clutter_ycb_configs"] = DataSource(
        source_type="task_assets",
        url="https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/pick_clutter/ycb_train_5k.json.gz",
        target_path="tasks/pick_clutter",
        checksum="70ec176c7036f326ea7813b77f8c03bea9db5960198498957a49b2895a9ec338",
    )
    DATA_SOURCES["assembling_kits"] = DataSource(
        source_type="task_assets",
        url="https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/assembling_kits_v1.zip",
        target_path="tasks/assembling_kits",
        checksum="e3371f17a07a012edaa3a0b3604fb1577f3fb921876c3d5ed59733dd75a6b4a0",
    )
    DATA_SOURCES["panda_avoid_obstacles"] = DataSource(
        source_type="task_assets",
        url="https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/avoid_obstacles/panda_train_2k.json.gz",
        target_path="tasks/avoid_obstacles",
        checksum="44dae9a0804172515c290c1f49a1e7e72d76e40201a2c5c7d4a3ccd43b4d5be4",
    )

    # ---------------------------------------------------------------------------- #
    # PartNet-mobility
    # ---------------------------------------------------------------------------- #
    category_uids = {}
    for category in ["cabinet_drawer", "cabinet_door", "chair", "bucket", "faucet"]:
        model_json = (
            PACKAGE_ASSET_DIR / f"partnet_mobility/meta/info_{category}_train.json"
        )
        model_ids = set(io_utils.load_json(model_json).keys())
        category_uids[category] = []
        for model_id in model_ids:
            uid = f"partnet_mobility/{model_id}"
            DATA_SOURCES[uid] = DataSource(
                source_type="objects",
                url=f"https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/partnet_mobility/dataset/{model_id}.zip",
                target_path=ASSET_DIR / "partnet_mobility" / "dataset" / model_id,
            )
            category_uids[category].append(uid)

    DATA_GROUPS["partnet_mobility_cabinet"] = set(
        category_uids["cabinet_drawer"] + category_uids["cabinet_door"]
    )
    DATA_GROUPS["partnet_mobility_chair"] = category_uids["chair"]
    DATA_GROUPS["partnet_mobility_bucket"] = category_uids["bucket"]
    DATA_GROUPS["partnet_mobility_faucet"] = category_uids["faucet"]
    DATA_GROUPS["partnet_mobility"] = set(
        category_uids["cabinet_drawer"]
        + category_uids["cabinet_door"]
        + category_uids["chair"]
        + category_uids["bucket"]
        + category_uids["faucet"]
    )

    # DATA_GROUPS["OpenCabinetDrawer-v1"] = category_uids["cabinet_drawer"]
    # DATA_GROUPS["OpenCabinetDoor-v1"] = category_uids["cabinet_door"]
    # DATA_GROUPS["PushChair-v1"] = category_uids["chair"]
    # DATA_GROUPS["MoveBucket-v1"] = category_uids["bucket"]
    # DATA_GROUPS["TurnFaucet-v1"] = category_uids["faucet"]

    # ---------------------------------------------------------------------------- #
    # Interactable Scene Datasets
    # ---------------------------------------------------------------------------- #
    DATA_SOURCES["ReplicaCAD"] = DataSource(
        source_type="scene",
        hf_repo_id="haosulab/ReplicaCAD",
        target_path="scene_datasets/replica_cad_dataset",
    )

    DATA_SOURCES["ReplicaCADRearrange"] = DataSource(
        source_type="scene",
        url="https://huggingface.co/datasets/haosulab/ReplicaCADRearrange/resolve/main/v1_extracted.zip",
        target_path="scene_datasets/replica_cad_dataset/rearrange",
    )

    DATA_SOURCES["AI2THOR"] = DataSource(
        source_type="scene",
        url="https://huggingface.co/datasets/haosulab/AI2THOR/resolve/main/ai2thor.zip",
        target_path="scene_datasets/ai2thor",
    )

    # Robots
    DATA_SOURCES["xmate3_robotiq"] = DataSource(
        source_type="robot",
        url="https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/xmate3_robotiq.zip",
        target_path="robots/xmate3_robotiq",
        checksum="ddda102a20eb41e28a0a501702e240e5d7f4084221a44f580e729f08b7c12d1a",
    )
    DATA_SOURCES["ur10e"] = DataSource(
        source_type="robot",
        url="https://github.com/haosulab/ManiSkill-UR10e/archive/refs/tags/v0.1.0.zip",
        target_path="robots/ur10e",
    )
    DATA_SOURCES["anymal_c"] = DataSource(
        source_type="robot",
        url="https://github.com/haosulab/ManiSkill-ANYmalC/archive/refs/tags/v0.1.1.zip",
        target_path="robots/anymal_c",
    )
    DATA_SOURCES["unitree_h1"] = DataSource(
        source_type="robot",
        url="https://github.com/haosulab/ManiSkill-UnitreeH1/archive/refs/tags/v0.1.0.zip",
        target_path="robots/unitree_h1",
    )
    DATA_SOURCES["unitree_g1"] = DataSource(
        source_type="robot",
        url="https://github.com/haosulab/ManiSkill-UnitreeG1/archive/refs/tags/v0.1.0.zip",
        target_path="robots/unitree_g1",
    )
    DATA_SOURCES["unitree_go2"] = DataSource(
        source_type="robot",
        url="https://github.com/haosulab/ManiSkill-UnitreeGo2/archive/refs/tags/v0.1.1.zip",
        target_path="robots/unitree_go2",
    )
    DATA_SOURCES["stompy"] = DataSource(
        source_type="robot",
        url="https://github.com/haosulab/ManiSkill-Stompy/archive/refs/tags/v0.1.0.zip",
        target_path="robots/stompy",
    )
    DATA_SOURCES["widowx250s"] = DataSource(
        source_type="robot",
        url="https://github.com/haosulab/ManiSkill-WidowX250S/archive/refs/tags/v0.1.0.zip",
        target_path="robots/widowx",
    )
    DATA_SOURCES["googlerobot"] = DataSource(
        source_type="robot",
        url="https://github.com/haosulab/ManiSkill-GoogleRobot/archive/refs/tags/v0.1.0.zip",
        target_path="robots/googlerobot",
    )
    DATA_SOURCES["robotiq_2f"] = DataSource(
        source_type="robot",
        url="https://github.com/haosulab/ManiSkill-Robotiq_2F/archive/refs/tags/v0.1.0.zip",
        target_path="robots/robotiq_2f",
    )


def expand_data_group_into_individual_data_source_ids(data_group_id: str):
    """Expand a data group into a list of individual data source IDs"""
    uids = []

    def helper(uid):
        nonlocal uids
        if uid in DATA_SOURCES:
            uids.append(uid)
        elif uid in DATA_GROUPS:
            [helper(x) for x in DATA_GROUPS[uid]]

    for uid in DATA_GROUPS[data_group_id]:
        helper(uid)
    uids = list(set(uids))
    return uids


initialize_data_sources()
