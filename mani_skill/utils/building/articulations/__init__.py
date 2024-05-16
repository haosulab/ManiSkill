from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.building.articulation_builder import ArticulationBuilder

from .robel import build_robel_valve


def get_articulation_builder(
    scene: ManiSkillScene,
    id: str,
    fix_root_link: bool = True,
    urdf_config: dict = dict(),
) -> ArticulationBuilder:
    """Builds an articulation or returns an articulation builder given an ID specifying which dataset/source and then the articulation ID

    Currently these IDs are hardcoded for a few datasets. The new Shapedex platform for hosting and managing all assets will be
    integrated in the future
    """
    splits = id.split(":")
    dataset_source = splits[0]
    articulation_id = ":".join(splits[1:])

    if dataset_source == "partnet-mobility":
        from .partnet_mobility import get_partnet_mobility_builder

        builder = get_partnet_mobility_builder(
            scene=scene,
            id=articulation_id,
            fix_root_link=fix_root_link,
            urdf_config=urdf_config,
        )
    else:
        raise RuntimeError(f"No dataset with id {dataset_source} was found")

    return builder
