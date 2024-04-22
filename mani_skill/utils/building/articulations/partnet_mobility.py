from mani_skill import ASSET_DIR, PACKAGE_ASSET_DIR
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.geometry.trimesh_utils import (
    get_articulation_meshes,
    merge_meshes,
)
from mani_skill.utils.io_utils import load_json

PARTNET_MOBILITY = None


def _load_partnet_mobility_dataset():
    global PARTNET_MOBILITY
    """loads preprocssed partnet mobility metadata"""
    PARTNET_MOBILITY = {
        "model_data": load_json(
            PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_cabinet_drawer_train.json"
        ),
    }

    def find_urdf_path(model_id):
        model_dir = ASSET_DIR / "partnet_mobility/dataset" / str(model_id)
        urdf_names = ["mobility_cvx.urdf", "mobility_fixed.urdf"]
        for urdf_name in urdf_names:
            urdf_path = model_dir / urdf_name
            if urdf_path.exists():
                return urdf_path

    PARTNET_MOBILITY["model_urdf_paths"] = {
        k: find_urdf_path(k) for k in PARTNET_MOBILITY["model_data"].keys()
    }


def get_partnet_mobility_builder(
    scene: ManiSkillScene, id: str, fix_root_link: bool = True
):
    global PARTNET_MOBILITY
    if PARTNET_MOBILITY is None:
        _load_partnet_mobility_dataset()
    metadata = PARTNET_MOBILITY["model_data"][id]
    loader = scene.create_urdf_loader()
    loader.fix_root_link = fix_root_link
    loader.scale = metadata["scale"]
    loader.load_multiple_collisions_from_file = True
    loader.disable_self_collisions = True
    urdf_path = PARTNET_MOBILITY["model_urdf_paths"][id]
    articulation_builders, _, _ = loader.parse(str(urdf_path))
    builder = articulation_builders[0]

    return builder
