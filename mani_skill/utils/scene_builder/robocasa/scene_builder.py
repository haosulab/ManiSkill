"""Implementation of the RoboCasa scene builder. Code ported from https://github.com/robocasa/robocasa"""

from typing import List

import torch
import yaml

from mani_skill.utils.scene_builder.robocasa.utils import scene_registry, scene_utils
from mani_skill.utils.scene_builder.scene_builder import SceneBuilder

FIXTURES = dict(wall=None)
# fixtures that are attached to other fixtures, disables positioning system in this script
FIXTURES_INTERIOR = dict(sink=None, stovetop=None, accessory=None, wall_accessory=None)

ALL_SIDES = ["left", "right", "front", "back", "bottom", "top"]


def check_syntax(fixture):
    """
    Checks that specifications of a fixture follows syntax rules
    """

    if fixture["type"] != "stack" and fixture["type"] not in FIXTURES:
        raise ValueError(
            'Invalid value for fixture type: "{}".'.format(fixture["type"])
        )

    if "config_name" in fixture and "default_config_name" in fixture:
        raise ValueError('Cannot specify both "config_name" and "default_config_name"')

    if "align_to" in fixture or "side" in fixture or "alignment" in fixture:
        if not ("align_to" in fixture and "side" in fixture):
            raise ValueError(
                'Both or neither of "align_to" and ' '"side" need to be specified.'
            )
        if "pos" in fixture:
            raise ValueError("Cannot specify both relative and absolute positions.")

        # check alignment and side arguments are compatible
        if "alignment" in fixture:
            for keywords in scene_utils.AXES_KEYWORDS.values():
                if fixture["side"] in keywords:
                    # check that neither keyword is used for alignment
                    if (
                        keywords[0] in fixture["alignment"]
                        or keywords[1] in fixture["alignment"]
                    ):
                        raise ValueError(
                            'Cannot set alignment to "{}" when aligning to the "{}" side'.format(
                                fixture["alignment"], fixture["side"]
                            )
                        )

        # check if side is valid
        if fixture["side"] not in ALL_SIDES:
            raise ValueError(
                '"{}" is not a valid side for alignment'.format(fixture["side"])
            )


class RoboCasaSceneBuilder(SceneBuilder):
    """
    SceneBuilder for the RoboCasa dataset: https://github.com/robocasa/robocasa

    TODO explain build config idxs and init config idxs
    """

    def build(self, build_config_idxs: List[int] = None):
        # return super().build(build_config_idxs)
        layout_path = scene_registry.get_layout_path(0)
        style_path = scene_registry.get_style_path(0)
        # load style
        with open(style_path, "r") as f:
            style = yaml.safe_load(f)

        # load arena
        with open(layout_path, "r") as f:
            arena_config = yaml.safe_load(f)

        # contains all fixtures with updated configs
        arena = list()

        # Update each fixture config. First iterate through groups: subparts of the arena that can be
        # rotated and displaced together. example: island group, right group, room group, etc
        for group_name, group_config in arena_config.items():
            group_fixtures = list()
            # each group is further divded into similar subcollections of fixtures
            # ex: main group counter accessories, main group top cabinets, etc
            for k, fixture_list in group_config.items():
                # these values are rotations/displacements that are applied to all fixtures in the group
                if k in ["group_origin", "group_z_rot", "group_pos"]:
                    continue
                elif type(fixture_list) != list:
                    raise ValueError(
                        '"{}" is not a valid argument for groups'.format(k)
                    )

                # add suffix to support different groups
                for fxtr_config in fixture_list:
                    fxtr_config["name"] += "_" + group_name
                    # update fixture names for alignment, interior objects, etc.
                    for k in scene_utils.ATTACH_ARGS + [
                        "align_to",
                        "stack_fixtures",
                        "size",
                    ]:
                        if k in fxtr_config:
                            if isinstance(fxtr_config[k], list):
                                for i in range(len(fxtr_config[k])):
                                    if isinstance(fxtr_config[k][i], str):
                                        fxtr_config[k][i] += "_" + group_name
                            else:
                                if isinstance(fxtr_config[k], str):
                                    fxtr_config[k] += "_" + group_name

                group_fixtures.extend(fixture_list)

            # update group rotation/displacement if necessary
            if "group_origin" in group_config:
                for fxtr_config in group_fixtures:
                    # do not update the rotation of the walls/floor
                    if fxtr_config["type"] in ["wall", "floor"]:
                        continue
                    fxtr_config["group_origin"] = group_config["group_origin"]
                    fxtr_config["group_pos"] = group_config["group_pos"]
                    fxtr_config["group_z_rot"] = group_config["group_z_rot"]

            # addto overall fixture list
            arena.extend(group_fixtures)

        # maps each fixture name to its object class
        fixtures = dict()
        # maps each fixture name to its configuration
        configs = dict()
        # names of composites, delete from fixtures before returning
        composites = list()
        # import ipdb;ipdb.set_trace()

        for fixture_config in arena:
            # scene_registry.check_syntax(fixture_config)
            fixture_name = fixture_config["name"]
            if "wall" not in fixture_name:
                continue
            # stack of fixtures, handled separately
            if fixture_config["type"] == "stack":
                stack = FixtureStack(
                    fixture_config,
                    fixtures,
                    configs,
                    style,
                    default_texture=None,
                    rng=rng,
                )
                fixtures[fixture_name] = stack
                configs[fixture_name] = fixture_config
                composites.append(fixture_name)
                continue

            # load style information and update config to include it
            default_config = scene_utils.load_style_config(style, fixture_config)
            if default_config is not None:
                for k, v in fixture_config.items():
                    default_config[k] = v
                fixture_config = default_config

            # set fixture type
            fixture_config["type"] = FIXTURES[fixture_config["type"]]

            # pre-processing for fixture size
            size = fixture_config.get("size", None)
            if isinstance(size, list):
                for i in range(len(size)):
                    elem = size[i]
                    if isinstance(elem, str):
                        ref_fxtr = fixtures[elem]
                        size[i] = ref_fxtr.size[i]

            # initialize fixture
            # TODO (stao): use batched episode rng later
            fixture = scene_utils.initialize_fixture(
                self.scene, fixture_config, fixtures, rng=self.env._episode_rng
            )
            fixtures[fixture_name] = fixture
            configs[fixture_name] = fixture_config

            # update fixture position
            if fixture_config["type"] not in FIXTURES_INTERIOR.values():
                # relative positioning
                if "align_to" in fixture_config:
                    pos = scene_utils.get_relative_position(
                        fixture,
                        fixture_config,
                        fixtures[fixture_config["align_to"]],
                        configs[fixture_config["align_to"]],
                    )

                elif "stack_on" in fixture_config:
                    stack_on = fixtures[fixture_config["stack_on"]]

                    # account for off-centered objects
                    stack_on_center = stack_on.center

                    # infer unspecified axes of position
                    pos = fixture_config["pos"]
                    if pos[0] is None:
                        pos[0] = stack_on.pos[0] + stack_on_center[0]
                    if pos[1] is None:
                        pos[1] = stack_on.pos[1] + stack_on_center[1]

                    # calculate height of fixture
                    pos[2] = (
                        stack_on.pos[2] + stack_on.size[2] / 2 + fixture.size[2] / 2
                    )
                    pos[2] += stack_on_center[2]
                else:
                    # absolute position
                    pos = fixture_config.get("pos", None)

                # TODO (stao): uncomment the next 2 lines
                # if pos is not None and type(fixture) not in [Wall, Floor]:
                #     fixture.set_pos(pos)

        # composites are non-MujocoObjects, must remove
        for composite in composites:
            del fixtures[composite]

    def initialize(self, env_idx: torch.Tensor, init_config_idxs: List[int] = None):
        pass