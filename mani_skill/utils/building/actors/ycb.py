from mani_skill import ASSET_DIR
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.io_utils import load_json

YCB_DATASET = dict()


def _load_ycb_dataset():
    global YCB_DATASET
    YCB_DATASET = {
        "model_data": load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json"),
    }


def get_ycb_builder(
    scene: ManiSkillScene, id: str, add_collision: bool = True, add_visual: bool = True
):
    if "YCB" not in YCB_DATASET:
        _load_ycb_dataset()
    model_db = YCB_DATASET["model_data"]

    builder = scene.create_actor_builder()

    metadata = model_db[id]
    density = metadata.get("density", 1000)
    model_scales = metadata.get("scales", [1.0])
    scale = model_scales[0]
    physical_material = None
    (metadata["bbox"]["max"][2] - metadata["bbox"]["min"][2]) * scale
    model_dir = ASSET_DIR / "assets/mani_skill2_ycb/models" / id
    if add_collision:
        collision_file = str(model_dir / "collision.ply")
        builder.add_multiple_convex_collisions_from_file(
            filename=collision_file,
            scale=[scale] * 3,
            material=physical_material,
            density=density,
        )
    if add_visual:
        visual_file = str(model_dir / "textured.obj")
        builder.add_visual_from_file(filename=visual_file, scale=[scale] * 3)

    return builder
