# second keyword corresponds to positive end of axis
from copy import deepcopy

import sapien
import yaml

from mani_skill import ASSET_DIR
from mani_skill.envs.scene import ManiSkillScene

ROBOCASA_ASSET_DIR = ASSET_DIR / "scene_datasets/robocasa_dataset/assets"

AXES_KEYWORDS = {0: ["left", "right"], 1: ["front", "back"], 2: ["bottom", "top"]}

# arguments not to be passed into fixture classes when initializing
IGNORE_ARGS = [
    "name",
    "align_to",
    "side",
    "alignment",
    "type",
    "center",
    "offset",
    "group_origin",
    "group_pos",
    "group_z_rot",
    "stack_height",
    "stack_fixtures",
]

# arguments used to point to other fixtures
ATTACH_ARGS = ["interior_obj", "stack_on", "attach_to"]


def initialize_fixture(scene: ManiSkillScene, config, cur_fixtures, rng=None):
    """
    initializes a fixture object based on the given configuration
    ignores positional arguments as it is changed later

    Args:
        config (dict): dictionary containing the fixture configuration.
                       Serves as the arguments to initialize the fixture

        cur_fixtures (dict): dictionary containing the current fixtures
    """

    config = deepcopy(config)
    name, class_type = config["name"], config["type"]

    # set size if stack_height is specified:
    if config.get("stack_height", None) is not None:
        stack_height = config["stack_height"]
        stack_fixtures = config["stack_fixtures"]
        curr_height = 0
        for fxtr in stack_fixtures:
            curr_height += cur_fixtures[fxtr].size[2]
        config["size"][2] = stack_height - curr_height

    # these fields should not be passed in when initializing the fixture
    for field in IGNORE_ARGS:
        if field in config:
            del config[field]

    if "pos" not in config:
        # need position to initialize fixture, adjusted later fo relative positioning
        config["pos"] = [0.0, 0.0, 0.0]
    # update fixture pointers
    for k in ATTACH_ARGS:
        if k in config:
            config[k] = cur_fixtures[config[k]]

    config["rng"] = rng
    fixture = class_type(scene=scene, name=name, **config)
    return fixture


def load_style_config(style, fixture_config):
    """
    Loads the style information for a given fixture. Style information can consist of
    which xml to use if there are multiple instances of a fixture, which texture to apply,
    which subcomponents to apply (panels/handles for a cab), etc.

    Args:
        style (dict): dictionary containing the style information for each fixture type

        fixture_config (dict): dictionary containing the fixture configuration
    """
    # accounts for the different types of cabinets
    fixture_type = fixture_config["type"]

    # cabinets, shelves, drawers, and boxes use the same default configurations
    if "cabinet" in fixture_type or "drawer" in fixture_type or "box" in fixture_type:
        fixture_type = "cabinet"

    # if fixture_type not in style:
    #     raise ValueError("Did not specify fixture type \"{}\" in chosen style".format(fixture_type))
    fixture_style = style.get(fixture_type, "default")

    yaml_path = str(
        ASSET_DIR
        / f"scene_datasets/robocasa_dataset/assets/fixtures/fixture_registry/{fixture_type}.yaml"
    )
    with open(yaml_path, "r") as f:
        default_configs = yaml.safe_load(f)

    # find which configuration to use
    if type(fixture_style) == dict and "config_name" not in fixture_config:
        if "default_config_name" in fixture_config:
            config_ids = fixture_style.get(
                fixture_config["default_config_name"], fixture_style["default"]
            )
            del fixture_config["default_config_name"]
        else:
            config_ids = fixture_style["default"]
    elif "default_config_name" in fixture_config:
        raise ValueError('Specified "default_config_name" but no default config found')

    elif "config_name" in fixture_config:
        config_ids = fixture_config["config_name"]
        del fixture_config["config_name"]
    else:
        config_ids = fixture_style

    # search for config by name
    config = default_configs["default"]
    if not isinstance(config_ids, list):
        config_ids = [config_ids]
    for cfg_id in config_ids:
        if cfg_id in default_configs:
            # add additional arguments based on default config
            additional_config = default_configs[cfg_id]
            for k, v in additional_config.items():
                config[k] = v
        else:
            raise ValueError(
                'Did not find style that matches "{}" for '
                'fixture type "{}"'.format(cfg_id, fixture_type)
            )
    return config


def get_relative_position(fixture, config, prev_fxtr, prev_fxtr_config):
    """
    Calculates the position of `fixture` based on a specified side
    and alignment relative to `prev_fixture`

    This assumes that the fixtures are properly centered!
    """

    side = config["side"].lower()
    alignment = config["alignment"].lower() if "alignment" in config else "center"
    size = fixture.size
    prev_pos, prev_size = deepcopy(prev_fxtr.pos), deepcopy(prev_fxtr.size)
    # for fixtures that are not perfectly centered (e.g. stoves)
    prev_pos += prev_fxtr.origin_offset

    # place fixtures next to each others
    for axis, keywords in AXES_KEYWORDS.items():
        if side not in keywords:
            continue
        pos = deepcopy(prev_pos)
        if side == keywords[1]:
            pos[axis] = prev_pos[axis] + prev_size[axis] / 2 + size[axis] / 2
        elif side == keywords[0]:
            pos[axis] = prev_pos[axis] - prev_size[axis] / 2 - size[axis] / 2

    # align such that the specified faces are flush
    # alignment - side compatibility is checked in check_syntax()
    for axis, keywords in AXES_KEYWORDS.items():
        if keywords[0] in alignment:
            pos[axis] = prev_pos[axis] - prev_size[axis] / 2 + size[axis] / 2
        elif keywords[1] in alignment:
            pos[axis] = prev_pos[axis] + prev_size[axis] / 2 - size[axis] / 2

    if "offset" in config:
        pos += config["offset"]

    # for fixtures that are not perfectly centered (e.g. stoves)
    pos -= fixture.origin_offset
    return pos
