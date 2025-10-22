"""Implementation of the RoboCasa scene builder. Code ported from https://github.com/robocasa/robocasa"""

import logging
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import yaml
from transforms3d.euler import euler2quat

from mani_skill.utils.scene_builder.robocasa.fixtures.accessories import (
    Accessory,
    CoffeeMachine,
    Stool,
    Toaster,
    WallAccessory,
)
from mani_skill.utils.scene_builder.robocasa.fixtures.cabinet import (
    Drawer,
    HingeCabinet,
    HousingCabinet,
    OpenCabinet,
    PanelCabinet,
    SingleCabinet,
)
from mani_skill.utils.scene_builder.robocasa.fixtures.counter import Counter
from mani_skill.utils.scene_builder.robocasa.fixtures.dishwasher import Dishwasher
from mani_skill.utils.scene_builder.robocasa.fixtures.fixture import (
    Fixture,
    FixtureType,
)
from mani_skill.utils.scene_builder.robocasa.fixtures.fixture_stack import FixtureStack
from mani_skill.utils.scene_builder.robocasa.fixtures.fixture_utils import (
    fixture_is_type,
)
from mani_skill.utils.scene_builder.robocasa.fixtures.fridge import Fridge
from mani_skill.utils.scene_builder.robocasa.fixtures.hood import Hood
from mani_skill.utils.scene_builder.robocasa.fixtures.microwave import Microwave
from mani_skill.utils.scene_builder.robocasa.fixtures.others import Box, Floor, Wall
from mani_skill.utils.scene_builder.robocasa.fixtures.sink import Sink
from mani_skill.utils.scene_builder.robocasa.fixtures.stove import Oven, Stove, Stovetop
from mani_skill.utils.scene_builder.robocasa.fixtures.windows import (
    FramedWindow,
    Window,
)
from mani_skill.utils.scene_builder.robocasa.utils import object_utils as OU
from mani_skill.utils.scene_builder.robocasa.utils import scene_registry, scene_utils
from mani_skill.utils.scene_builder.robocasa.utils.placement_samplers import (
    RandomizationError,
    SequentialCompositeSampler,
    UniformRandomSampler,
)
from mani_skill.utils.scene_builder.scene_builder import SceneBuilder
from mani_skill.utils.structs import Actor
from mani_skill.utils.structs.pose import Pose

FIXTURES = dict(
    hinge_cabinet=HingeCabinet,
    single_cabinet=SingleCabinet,
    open_cabinet=OpenCabinet,
    panel_cabinet=PanelCabinet,
    housing_cabinet=HousingCabinet,
    drawer=Drawer,
    counter=Counter,
    stove=Stove,
    stovetop=Stovetop,
    oven=Oven,
    microwave=Microwave,
    hood=Hood,
    sink=Sink,
    fridge=Fridge,
    dishwasher=Dishwasher,
    wall=Wall,
    floor=Floor,
    box=Box,
    accessory=Accessory,
    paper_towel=Accessory,
    plant=Accessory,
    knife_block=Accessory,
    stool=Stool,
    utensil_holder=Accessory,
    coffee_machine=CoffeeMachine,
    toaster=Toaster,
    utensil_rack=WallAccessory,
    wall_accessory=WallAccessory,
    window=Window,
    framed_window=FramedWindow,
    # needs some additional work
    # slide_cabinet=SlideCabinet,
)
# fixtures that are attached to other fixtures, disables positioning system in this script
FIXTURES_INTERIOR = dict(
    sink=Sink, stovetop=Stovetop, accessory=Accessory, wall_accessory=WallAccessory
)

ALL_SIDES = ["left", "right", "front", "back", "bottom", "top"]

ROBOT_FRONT_FACING_SIZE = dict(fetch=0.8, unitree_g1_simplified_upper_body=0.6)


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

    def __init__(self, *args, init_robot_base_pos=None, **kwargs):
        self.init_robot_base_pos = init_robot_base_pos
        self.scene_data = []  # maps scene_idx to {"fixtures", "fxtr_placements"}
        super().__init__(*args, **kwargs)

    def build(self, build_config_idxs: Optional[list[int]] = None):
        if self.env.agent is not None:
            self.robot_poses = self.env.agent.robot.initial_pose
        else:
            self.robot_poses = None
        if build_config_idxs is None:
            build_config_idxs = []
            for i in range(self.env.num_envs):
                # Total number of configs is 10 * 12 = 120
                config_idx = self.env._batched_episode_rng[i].randint(0, 120)
                build_config_idxs.append(config_idx)

        for scene_idx, build_config_idx in enumerate(build_config_idxs):
            layout_idx = build_config_idx // 12  # Get layout index (0-9)
            style_idx = build_config_idx % 12  # Get style index (0-11)
            layout_path = scene_registry.get_layout_path(layout_idx)
            style_path = scene_registry.get_style_path(style_idx)
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
            fixtures: dict[str, Fixture] = dict()
            # maps each fixture name to its configuration
            configs = dict()
            # names of composites, delete from fixtures before returning
            composites = list()

            for fixture_config in arena:
                # scene_registry.check_syntax(fixture_config)
                fixture_name = fixture_config["name"]

                # stack of fixtures, handled separately
                if fixture_config["type"] == "stack":
                    stack = FixtureStack(
                        self.scene,
                        fixture_config,
                        fixtures,
                        configs,
                        style,
                        default_texture=None,
                        rng=self.env._batched_episode_rng[scene_idx],
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
                if fixture_config["type"] not in FIXTURES:
                    continue
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
                    self.scene,
                    fixture_config,
                    fixtures,
                    rng=self.env._batched_episode_rng[scene_idx],
                )

                fixtures[fixture_name] = fixture
                configs[fixture_name] = fixture_config
                pos = None
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
                if pos is not None and type(fixture) not in [Wall, Floor]:
                    fixture.set_pos(deepcopy(pos))
            # composites are non-MujocoObjects, must remove
            for composite in composites:
                del fixtures[composite]

            # update the rotation and postion of each fixture based on their group
            for name, fixture in fixtures.items():
                # check if updates are necessary
                config = configs[name]
                if "group_origin" not in config:
                    continue

                # TODO: add default for group origin?
                # rotate about this coordinate (around the z-axis)
                origin = config["group_origin"]
                pos = config["group_pos"]
                z_rot = config["group_z_rot"]
                displacement = [pos[0] - origin[0], pos[1] - origin[1]]

                if type(fixture) not in [Wall, Floor]:
                    dx = fixture.pos[0] - origin[0]
                    dy = fixture.pos[1] - origin[1]
                    dx_rot = dx * np.cos(z_rot) - dy * np.sin(z_rot)
                    dy_rot = dx * np.sin(z_rot) + dy * np.cos(z_rot)

                    x_rot = origin[0] + dx_rot
                    y_rot = origin[1] + dy_rot
                    z = fixture.pos[2]
                    pos_new = [x_rot + displacement[0], y_rot + displacement[1], z]

                    # account for previous z-axis rotation
                    rot_prev = fixture.euler
                    if rot_prev is not None:
                        # TODO: switch to quaternion since euler rotations are ambiguous
                        rot_new = rot_prev
                        rot_new[2] += z_rot
                    else:
                        rot_new = [0, 0, z_rot]
                    fixture.pos = np.array(pos_new)
                    fixture.set_euler(rot_new)

            # self.actors = actors
            # fixtures = fixtures
            fixture_cfgs = self.get_fixture_cfgs(fixtures)
            # generate initial poses for objects so that they are spawned in nice places during GPU initialization
            # to be more performant
            (
                fxtr_placements,
                robot_base_pos,
                robot_base_ori,
            ) = self._generate_initial_placements(
                fixtures, fixture_cfgs, rng=self.env._batched_episode_rng[scene_idx]
            )
            self.scene_data.append(
                dict(
                    fixtures=fixtures,
                    fxtr_placements=fxtr_placements,
                    fixture_cfgs=fixture_cfgs,
                )
            )

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in fxtr_placements.values():
                assert isinstance(obj, Fixture)
                obj.pos = obj_pos
                obj.quat = obj_quat

            if self.env.agent is not None:
                self.robot_poses.raw_pose[scene_idx][:3] = torch.from_numpy(
                    robot_base_pos
                ).to(self.robot_poses.device)
                self.robot_poses.raw_pose[scene_idx][3:] = torch.from_numpy(
                    euler2quat(*robot_base_ori)
                ).to(self.robot_poses.device)

            actors: dict[str, Actor] = {}

            ### collision handling and optimization ###
            # Generally we aim to ensure all articulations in a stack have the same collision bits so they can't collide with each other
            # and with a range of [22, 30] we can generally ensure adjacent articulations can collide with each other.
            # walls and floors cannot collide with anything. Walls can only collide with the robot. They are assigned bits 22 to 30.
            # mobile base robots have their wheels/non base links assigned bit of 30 to not collide with the floor or walls.
            # the base links can optionally be also assigned a bit of 31 to not collide with walls.

            # fixtures that are not articulated are always static and cannot hit other non-articulated fixtures. This scenario is assigned bit 21.
            actor_bit = 21
            # prismatic_drawer_bit = 25

            collision_start_bit = 22
            fixture_idx = 0
            stack_collision_bits = dict()
            for stack_index, stack in enumerate(composites):
                stack_collision_bits[stack] = collision_start_bit + stack_index % 9
            for k, v in fixtures.items():
                fixture_idx += 1
                built = v.build(scene_idxs=[scene_idx])
                if built is not None:
                    actors[k] = built
                    # ensure all rooted articulated objects have collisions ignored with all static objects
                    # ensure all articulations in the same stack have the same collision bits, since by definition for robocasa they cannot
                    # collide with each other
                    if (
                        built.is_articulation
                        and built.articulation.fixed_root_link.all()
                    ):
                        collision_bit = collision_start_bit + fixture_idx % 5
                        if "stack" in v.name:
                            for stack_group in stack_collision_bits.keys():
                                if stack_group in v.name:
                                    collision_bit = stack_collision_bits[stack_group]
                                    break
                        # is_prismatic_cabinet = False
                        # for joint in built.articulation.joints:
                        #     if joint.type[0] == "prismatic":
                        #         is_prismatic_cabinet = True
                        #         break
                        for link in built.articulation.links:
                            # if "object" in link.name:
                            #     import ipdb; ipdb.set_trace()
                            link.set_collision_group(
                                group=2, value=0
                            )  # clear all default ignored collisions
                            if link.joint.type[0] == "fixed":
                                link.set_collision_group_bit(
                                    group=2, bit_idx=actor_bit, bit=1
                                )
                            link.set_collision_group_bit(
                                group=2, bit_idx=collision_bit, bit=1
                            )

                    else:
                        if built.actor.px_body_type == "static":
                            collision_bit = collision_start_bit + fixture_idx % 5
                            if "stack" in v.name:
                                for stack_group in stack_collision_bits.keys():
                                    if stack_group in v.name:
                                        collision_bit = stack_collision_bits[
                                            stack_group
                                        ]
                                        break
                            if isinstance(v, Floor):
                                for bit_idx in range(21, 32):
                                    built.actor.set_collision_group_bit(
                                        group=2, bit_idx=bit_idx, bit=1
                                    )
                            elif isinstance(v, Wall):
                                for bit_idx in range(21, 31):
                                    built.actor.set_collision_group_bit(
                                        group=2, bit_idx=bit_idx, bit=1
                                    )

                            else:
                                built.actor.set_collision_group_bit(
                                    group=2,
                                    bit_idx=collision_bit,
                                    bit=1,
                                )
                                built.actor.set_collision_group_bit(
                                    group=2, bit_idx=actor_bit, bit=1
                                )
            # self.actors = actors

        # disable collisions

        if self.env.robot_uids == "fetch":
            self.env.agent
            for link in [self.env.agent.l_wheel_link, self.env.agent.r_wheel_link]:
                for bit_idx in range(25, 31):
                    link.set_collision_group_bit(group=2, bit_idx=bit_idx, bit=1)
            # for bit_idx in range(25, 31):
            self.env.agent.base_link.set_collision_group_bit(group=2, bit_idx=31, bit=1)

        elif self.env.robot_uids == "unitree_g1_simplified_upper_body":
            # TODO (stao): determine collisions to disable for unitree robot
            pass

    def _generate_initial_placements(
        self, fixtures, fixture_cfgs, rng: np.random.RandomState
    ):
        """Generate and places randomized fixtures and robot(s) into the scene. This code is not parallelized"""
        fxtr_placement_initializer = self._get_placement_initializer(
            fixtures, dict(), fixture_cfgs, z_offset=0.0, rng=rng
        )
        fxtr_placements = None
        for i in range(10):
            try:
                fxtr_placements = fxtr_placement_initializer.sample()
            except RandomizationError:
                # if macros.VERBOSE:
                #     print("Ranomization error in initial placement. Try #{}".format(i))
                continue
            break
        if fxtr_placements is None:
            # if macros.VERBOSE:
            # print("Could not place fixtures.")
            # self._load_model()
            raise RuntimeError("Could not place fixtures.")

        # setup internal references related to fixtures
        # self._setup_kitchen_references()

        # set robot position
        if self.init_robot_base_pos is not None:
            ref_fixture = self.get_fixture(fixtures, self.init_robot_base_pos)
        else:
            valid_src_fixture_classes = [
                "CoffeeMachine",
                "Toaster",
                "Stove",
                "Stovetop",
                "SingleCabinet",
                "HingeCabinet",
                "OpenCabinet",
                "Drawer",
                "Microwave",
                "Sink",
                "Hood",
                "Oven",
                "Fridge",
                "Dishwasher",
            ]
            while True:
                ref_fixture = rng.choice(list(fixtures.values()))
                fxtr_class = type(ref_fixture).__name__
                if fxtr_class not in valid_src_fixture_classes:
                    continue
                break

        if self.env.agent is not None:
            robot_base_pos, robot_base_ori = self.compute_robot_base_placement_pose(
                fixtures, ref_fixture
            )
        else:
            robot_base_pos = None
            robot_base_ori = None
        return fxtr_placements, robot_base_pos, robot_base_ori

    def initialize(self, env_idx: torch.Tensor, init_config_idxs: list[int] = None):
        with torch.device(self.env.device):
            if self.env.agent is not None:
                if self.env.robot_uids == "fetch":
                    self.env.agent.robot.set_qpos(self.env.agent.keyframes["rest"].qpos)
                    self.env.agent.robot.set_pose(self.robot_poses[env_idx])
                elif self.env.robot_uids == "unitree_g1_simplified_upper_body":
                    self.env.agent.robot.set_qpos(
                        self.env.agent.keyframes["standing"].qpos
                    )
                    xyz = self.env.agent.robot.pose.p
                    xyz[:, 2] += self.env.agent.keyframes["standing"].pose.p[2]
                    self.env.agent.robot.set_pose(
                        Pose.create_from_pq(p=xyz, q=self.env.agent.robot.pose.q)
                    )

    def get_fixture_cfgs(self, fixtures):
        """
        Returns config data for all fixtures in the arena

        Returns:
            list: list of fixture configurations
        """
        fixture_cfgs = []
        for (name, fxtr) in fixtures.items():
            cfg = {}
            cfg["name"] = name
            cfg["model"] = fxtr
            cfg["type"] = "fixture"
            if hasattr(fxtr, "_placement"):
                cfg["placement"] = fxtr._placement

            fixture_cfgs.append(cfg)

        return fixture_cfgs

    def _is_fxtr_valid(self, fxtr, size):
        """
        checks if counter is valid for object placement by making sure it is large enough

        Args:
            fxtr (Fixture): fixture to check
            size (tuple): minimum size (x,y) that the counter region must be to be valid

        Returns:
            bool: True if fixture is valid, False otherwise
        """
        return True
        for region in fxtr.get_reset_regions(self).values():
            if region["size"][0] >= size[0] and region["size"][1] >= size[1]:
                return True
        return False

    def get_fixture(self, fixtures, id, ref=None, size=(0.2, 0.2)):
        """
        search fixture by id (name, object, or type)

        Args:
            id (str, Fixture, FixtureType): id of fixture to search for

            ref (str, Fixture, FixtureType): if specified, will search for fixture close to ref (within 0.10m)

            size (tuple): if sampling counter, minimum size (x,y) that the counter region must be

        Returns:
            Fixture: fixture object
        """
        # case 1: id refers to fixture object directly
        if isinstance(id, Fixture):
            return id
        # case 2: id refers to exact name of fixture
        elif id in fixtures.keys():
            return fixtures[id]

        if ref is None:
            # find all fixtures with names containing given name
            if isinstance(id, FixtureType) or isinstance(id, int):
                matches = [
                    name
                    for (name, fxtr) in fixtures.items()
                    if fixture_is_type(fxtr, id)
                ]
            else:
                matches = [name for name in fixtures.keys() if id in name]
            if id == FixtureType.COUNTER or id == FixtureType.COUNTER_NON_CORNER:
                matches = [
                    name
                    for name in matches
                    if self._is_fxtr_valid(fixtures[name], size)
                ]
            assert len(matches) > 0
            # sample random key
            # TODO (stao): fix the key!
            key = self.env._episode_rng.choice(matches)
            return fixtures[key]
        else:
            ref_fixture = self.get_fixture(fixtures, ref)

            assert isinstance(id, FixtureType)
            cand_fixtures = []
            for fxtr in fixtures.values():
                if not fixture_is_type(fxtr, id):
                    continue
                if fxtr is ref_fixture:
                    continue
                if id == FixtureType.COUNTER:
                    fxtr_is_valid = self._is_fxtr_valid(fxtr, size)
                    if not fxtr_is_valid:
                        continue
                cand_fixtures.append(fxtr)

            # first, try to find fixture "containing" the reference fixture
            for fxtr in cand_fixtures:
                if OU.point_in_fixture(ref_fixture.pos, fxtr, only_2d=True):
                    return fxtr
            # if no fixture contains reference fixture, sample all close fixtures
            dists = [
                OU.fixture_pairwise_dist(ref_fixture, fxtr) for fxtr in cand_fixtures
            ]
            min_dist = np.min(dists)
            close_fixtures = [
                fxtr for (fxtr, d) in zip(cand_fixtures, dists) if d - min_dist < 0.10
            ]
            return self.rng.choice(close_fixtures)

    def compute_robot_base_placement_pose(self, fixtures, ref_fixture, offset=None):
        """
        steps:
        1. find the nearest counter to this fixture
        2. compute offset relative to this counter
        3. transform offset to global coordinates

        Args:
            ref_fixture (Fixture): reference fixture to place th robot near

            offset (list): offset to add to the base position

        """
        # step 1: find vase fixture closest to robot
        base_fixture = None

        # get all base fixtures in the environment
        base_fixtures = [
            fxtr
            for fxtr in fixtures.values()
            if isinstance(fxtr, Counter)
            or isinstance(fxtr, Stove)
            or isinstance(fxtr, Stovetop)
            or isinstance(fxtr, HousingCabinet)
            or isinstance(fxtr, Fridge)
        ]

        for fxtr in base_fixtures:
            # get bounds of fixture
            point = ref_fixture.pos
            if not OU.point_in_fixture(point=point, fixture=fxtr, only_2d=True):
                continue
            base_fixture = fxtr
            break

        # set the base fixture as the ref fixture itself if cannot find fixture containing ref
        if base_fixture is None:
            base_fixture = ref_fixture
        # assert base_fixture is not None

        # step 2: compute offset relative to this counter
        base_to_ref, _ = OU.get_rel_transform(base_fixture, ref_fixture)
        cntr_y = base_fixture.get_ext_sites(relative=True)[0][1]
        if self.env.agent.robot.name in ROBOT_FRONT_FACING_SIZE:
            front_face_size = ROBOT_FRONT_FACING_SIZE[self.env.agent.robot.name]
        else:
            logging.warning(
                f"Robot {self.env.agent.robot.name} doesn't have a defined front facing size, defaulting to 0.7m"
            )
            front_face_size = 0.7
        base_to_edge = [
            base_to_ref[0],
            cntr_y - front_face_size,
            0,
        ]
        if offset is not None:
            base_to_edge[0] += offset[0]
            base_to_edge[1] += offset[1]

        if (
            isinstance(base_fixture, HousingCabinet)
            or isinstance(base_fixture, Fridge)
            or "stack" in base_fixture.name
        ):
            base_to_edge[1] -= 0.10

        # step 3: transform offset to global coordinates
        robot_base_pos = np.zeros(3)
        robot_base_pos[0:2] = OU.get_pos_after_rel_offset(base_fixture, base_to_edge)[
            0:2
        ]
        robot_base_ori = np.array([0, 0, base_fixture.rot + np.pi / 2])

        return robot_base_pos, robot_base_ori

    def _get_placement_initializer(
        self,
        fixtures,
        objects,
        cfg_list,
        z_offset=0.01,
        rng: np.random.RandomState = None,
    ):

        """
        Creates a placement initializer for the objects/fixtures based on the specifications in the configurations list

        Args:
            cfg_list (list): list of object configurations

            z_offset (float): offset in z direction

        Returns:
            SequentialCompositeSampler: placement initializer

        """

        placement_initializer = SequentialCompositeSampler(name="SceneSampler", rng=rng)

        for (obj_i, cfg) in enumerate(cfg_list):
            # determine which object is being placed
            if cfg["type"] == "fixture":
                mj_obj = fixtures[cfg["name"]]
            elif cfg["type"] == "object":
                mj_obj = objects[cfg["name"]]
            else:
                raise ValueError
            placement = cfg.get("placement", None)
            if placement is None:
                continue
            fixture_id = placement.get("fixture", None)
            if fixture_id is not None:
                # get fixture to place object on
                fixture = self.get_fixture(
                    fixtures,
                    id=fixture_id,
                    ref=placement.get("ref", None),
                )

                # calculate the total available space where object could be placed
                sample_region_kwargs = placement.get("sample_region_kwargs", {})

                reset_region = fixture.sample_reset_region(
                    env=self, fixtures=fixtures, **sample_region_kwargs
                )
                outer_size = reset_region["size"]
                margin = placement.get("margin", 0.04)
                outer_size = (outer_size[0] - margin, outer_size[1] - margin)

                # calculate the size of the inner region where object will actually be placed
                target_size = placement.get("size", None)
                if target_size is not None:
                    target_size = deepcopy(list(target_size))
                    for size_dim in [0, 1]:
                        if target_size[size_dim] == "obj":
                            target_size[size_dim] = mj_obj.size[size_dim] + 0.005
                        if target_size[size_dim] == "obj.x":
                            target_size[size_dim] = mj_obj.size[0] + 0.005
                        if target_size[size_dim] == "obj.y":
                            target_size[size_dim] = mj_obj.size[1] + 0.005
                    inner_size = np.min((outer_size, target_size), axis=0)
                else:
                    inner_size = outer_size

                inner_xpos, inner_ypos = placement.get("pos", (None, None))
                offset = placement.get("offset", (0.0, 0.0))

                # center inner region within outer region
                if inner_xpos == "ref":
                    # compute optimal placement of inner region to match up with the reference fixture
                    x_halfsize = outer_size[0] / 2 - inner_size[0] / 2
                    if x_halfsize == 0.0:
                        inner_xpos = 0.0
                    else:
                        ref_fixture = self.get_fixture(
                            fixtures, placement["sample_region_kwargs"]["ref"]
                        )
                        ref_pos = ref_fixture.pos
                        fixture_to_ref = OU.get_rel_transform(fixture, ref_fixture)[0]
                        outer_to_ref = fixture_to_ref - reset_region["offset"]
                        inner_xpos = outer_to_ref[0] / x_halfsize
                        inner_xpos = np.clip(inner_xpos, a_min=-1.0, a_max=1.0)
                elif inner_xpos is None:
                    inner_xpos = 0.0

                if inner_ypos is None:
                    inner_ypos = 0.0
                # offset for inner region
                intra_offset = (
                    (outer_size[0] / 2 - inner_size[0] / 2) * inner_xpos + offset[0],
                    (outer_size[1] / 2 - inner_size[1] / 2) * inner_ypos + offset[1],
                )
                # center surface point of entire region
                ref_pos = fixture.pos + [0, 0, reset_region["offset"][2]]
                ref_rot = fixture.rot

                # x, y, and rotational ranges for randomization
                x_range = (
                    np.array([-inner_size[0] / 2, inner_size[0] / 2])
                    + reset_region["offset"][0]
                    + intra_offset[0]
                )
                y_range = (
                    np.array([-inner_size[1] / 2, inner_size[1] / 2])
                    + reset_region["offset"][1]
                    + intra_offset[1]
                )
                rotation = placement.get("rotation", np.array([-np.pi / 4, np.pi / 4]))
            else:
                target_size = placement.get("size", None)
                x_range = np.array([-target_size[0] / 2, target_size[0] / 2])
                y_range = np.array([-target_size[1] / 2, target_size[1] / 2])
                rotation = placement.get("rotation", np.array([-np.pi / 4, np.pi / 4]))
                ref_pos = [0, 0, 0]
                ref_rot = 0.0

            # if macros.SHOW_SITES is True:
            #     """
            #     show outer reset region
            #     """
            #     pos_to_vis = deepcopy(ref_pos)
            #     pos_to_vis[:2] += T.rotate_2d_point(
            #         [reset_region["offset"][0], reset_region["offset"][1]], rot=ref_rot
            #     )
            #     size_to_vis = np.concatenate(
            #         [
            #             np.abs(
            #                 T.rotate_2d_point(
            #                     [outer_size[0] / 2, outer_size[1] / 2], rot=ref_rot
            #                 )
            #             ),
            #             [0.001],
            #         ]
            #     )
            #     site_str = """<site type="box" rgba="0 0 1 0.4" size="{size}" pos="{pos}" name="reset_region_outer_{postfix}"/>""".format(
            #         pos=array_to_string(pos_to_vis),
            #         size=array_to_string(size_to_vis),
            #         postfix=str(obj_i),
            #     )
            #     site_tree = ET.fromstring(site_str)
            #     self.model.worldbody.append(site_tree)

            #     """
            #     show inner reset region
            #     """
            #     pos_to_vis = deepcopy(ref_pos)
            #     pos_to_vis[:2] += T.rotate_2d_point(
            #         [np.mean(x_range), np.mean(y_range)], rot=ref_rot
            #     )
            #     size_to_vis = np.concatenate(
            #         [
            #             np.abs(
            #                 T.rotate_2d_point(
            #                     [
            #                         (x_range[1] - x_range[0]) / 2,
            #                         (y_range[1] - y_range[0]) / 2,
            #                     ],
            #                     rot=ref_rot,
            #                 )
            #             ),
            #             [0.002],
            #         ]
            #     )
            #     site_str = """<site type="box" rgba="1 0 0 0.4" size="{size}" pos="{pos}" name="reset_region_inner_{postfix}"/>""".format(
            #         pos=array_to_string(pos_to_vis),
            #         size=array_to_string(size_to_vis),
            #         postfix=str(obj_i),
            #     )
            #     site_tree = ET.fromstring(site_str)
            #     self.model.worldbody.append(site_tree)

            placement_initializer.append_sampler(
                sampler=UniformRandomSampler(
                    name="{}_Sampler".format(cfg["name"]),
                    mujoco_objects=mj_obj,
                    x_range=x_range,
                    y_range=y_range,
                    rotation=rotation,
                    ensure_object_boundary_in_range=placement.get(
                        "ensure_object_boundary_in_range", True
                    ),
                    ensure_valid_placement=placement.get(
                        "ensure_valid_placement", True
                    ),
                    reference_pos=ref_pos,
                    reference_rot=ref_rot,
                    z_offset=z_offset,
                    rng=rng,
                    rotation_axis=placement.get("rotation_axis", "z"),
                ),
                sample_args=placement.get("sample_args", None),
            )

        return placement_initializer
