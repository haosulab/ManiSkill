"""Code ported over from RoboCasa: https://github.com/robocasa/robocasa/blob/main/robocasa/models/scenes/scene_registry.py"""

from collections import OrderedDict
from enum import IntEnum

from mani_skill import ASSET_DIR
from mani_skill.utils.scene_builder.robocasa.utils.scene_utils import ROBOCASA_ASSET_DIR


class LayoutType(IntEnum):
    """
    Enum for available layouts in RoboCasa environment
    """

    ONE_WALL_SMALL = 0
    ONE_WALL_LARGE = 1
    L_SHAPED_SMALL = 2
    L_SHAPED_LARGE = 3
    GALLEY = 4
    U_SHAPED_SMALL = 5
    U_SHAPED_LARGE = 6
    G_SHAPED_SMALL = 7
    G_SHAPED_LARGE = 8
    WRAPAROUND = 9

    # negative values correspond to groups (see LAYOUT_GROUPS_TO_IDS)
    ALL = -1
    NO_ISLAND = -2
    ISLAND = -3
    DINING = -4


LAYOUT_GROUPS_TO_IDS = {
    -1: list(range(10)),  # all
    -2: [0, 2, 4, 5, 7],  # no island
    -3: [1, 3, 6, 8, 9],  # island
    -4: [1, 3, 6, 7, 8, 9],  # dining
}


class StyleType(IntEnum):
    """
    Enums for available styles in RoboCasa environment
    """

    INDUSTRIAL = 0
    SCANDANAVIAN = 1
    COASTAL = 2
    MODERN_1 = 3
    MODERN_2 = 4
    TRADITIONAL_1 = 5
    TRADITIONAL_2 = 6
    FARMHOUSE = 7
    RUSTIC = 8
    MEDITERRANEAN = 9
    TRANSITIONAL_1 = 10
    TRANSITIONAL_2 = 11

    # negative values correspond to groups
    ALL = -1


STYLE_GROUPS_TO_IDS = {
    -1: list(range(12)),  # all
}


def get_layout_path(layout_id):
    """
    Get corresponding blueprint filepath (yaml) for a layout

    Args:
        layout_id (int or LayoutType): layout id (int or enum)

    Return:
        str: yaml path for specified layout
    """
    if isinstance(layout_id, int):
        layout_int_to_name = dict(
            map(lambda item: (item.value, item.name.lower()), LayoutType)
        )
        layout_name = layout_int_to_name[layout_id]
    elif isinstance(layout_id, LayoutType):
        layout_name = layout_id.name.lower()
    else:
        raise ValueError

    # special case: if name starts with one letter, capitalize it
    if layout_name[1] == "_":
        layout_name = layout_name.capitalize()

    return str(ROBOCASA_ASSET_DIR / f"scenes/kitchen_layouts/{layout_name}.yaml")


def get_style_path(style_id):
    """
    Get corresponding blueprint filepath (yaml) for a style

    Args:
        style_id (int or StyleType): style id (int or enum)

    Return:
        str: yaml path for specified style
    """
    if isinstance(style_id, int):
        style_int_to_name = dict(
            map(lambda item: (item.value, item.name.lower()), StyleType)
        )
        style_name = style_int_to_name[style_id]
    elif isinstance(style_id, StyleType):
        style_name = style_id.name.lower()
    else:
        raise ValueError
    return (
        ASSET_DIR
        / f"scene_datasets/robocasa_dataset/assets/scenes/kitchen_styles/{style_name}.yaml"
    )


def unpack_layout_ids(layout_ids):
    if layout_ids is None:
        layout_ids = LayoutType.ALL

    if not isinstance(layout_ids, list):
        layout_ids = [layout_ids]

    layout_ids = [int(id) for id in layout_ids]

    all_layout_ids = []
    for id in layout_ids:
        if id < 0:
            all_layout_ids += LAYOUT_GROUPS_TO_IDS[id]
        else:
            all_layout_ids.append(id)
    return list(OrderedDict.fromkeys(all_layout_ids))


def unpack_style_ids(style_ids):
    if style_ids is None:
        style_ids = StyleType.ALL

    if not isinstance(style_ids, list):
        style_ids = [style_ids]

    style_ids = [int(id) for id in style_ids]

    all_style_ids = []
    for id in style_ids:
        if id < 0:
            all_style_ids += STYLE_GROUPS_TO_IDS[id]
        else:
            all_style_ids.append(id)
    return list(OrderedDict.fromkeys(all_style_ids))
