from mani_skill.utils.scene_builder.robocasa.fixtures.accessories import Stool, Toaster
from mani_skill.utils.scene_builder.robocasa.fixtures.cabinet import (
    Cabinet,
    Drawer,
    HingeCabinet,
    OpenCabinet,
    SingleCabinet,
)
from mani_skill.utils.scene_builder.robocasa.fixtures.counter import Counter
from mani_skill.utils.scene_builder.robocasa.fixtures.fixture import FixtureType
from mani_skill.utils.scene_builder.robocasa.fixtures.microwave import Microwave
from mani_skill.utils.scene_builder.robocasa.fixtures.sink import Sink
from mani_skill.utils.scene_builder.robocasa.fixtures.stove import Stove


def fixture_is_type(fixture, fixture_type):
    """
    Check if a fixture is of a certain type

    Args:
        fixture (Fixture): The fixture to check

        fixture_type (FixtureType): The type to check against
    """
    if fixture_type == FixtureType.COUNTER:
        return isinstance(fixture, Counter)
    elif fixture_type == FixtureType.DINING_COUNTER:
        cls_check = any([isinstance(fixture, cls) for cls in [Counter]])
        if not cls_check:
            return False
        return fixture.width >= 2 or fixture.depth >= 2
    elif fixture_type == FixtureType.CABINET:
        return isinstance(fixture, Cabinet)
    elif fixture_type == FixtureType.DRAWER:
        return isinstance(fixture, Drawer)
    elif fixture_type == FixtureType.SINK:
        return isinstance(fixture, Sink)
    elif fixture_type == FixtureType.STOVE:
        return isinstance(fixture, Stove)
    elif fixture_type == FixtureType.CABINET_TOP:
        cls_check = any(
            [
                isinstance(fixture, cls)
                for cls in [SingleCabinet, HingeCabinet, OpenCabinet]
            ]
        )
        if not cls_check:
            return False
        if "stack" in fixture.name:  # wall stack cabinets not valid
            return False
        # check the height of the cabinet to see if it is a top cabinet
        fxtr_bottom_z = fixture.pos[2] + fixture.bottom_offset[2]
        height_check = 1.0 <= fxtr_bottom_z <= 1.60
        return height_check
    elif fixture_type == FixtureType.MICROWAVE:
        return isinstance(fixture, Microwave)
    elif fixture_type in [FixtureType.DOOR_HINGE, FixtureType.DOOR_TOP_HINGE]:
        cls_check = any(
            [
                isinstance(fixture, cls)
                for cls in [SingleCabinet, HingeCabinet, Microwave]
            ]
        )
        if not cls_check:
            return False
        if fixture_type == FixtureType.DOOR_TOP_HINGE:
            if "stack" in fixture.name:  # wall stack cabinets not valid
                return False
            fxtr_bottom_z = fixture.pos[2] + fixture.bottom_offset[2]
            height_check = 1.0 <= fxtr_bottom_z <= 1.60
            if not height_check:
                return False
        return True
    elif fixture_type in [
        FixtureType.DOOR_HINGE_SINGLE,
        FixtureType.DOOR_TOP_HINGE_SINGLE,
    ]:
        cls_check = any(
            [isinstance(fixture, cls) for cls in [SingleCabinet, Microwave]]
        )
        if not cls_check:
            return False
        if fixture_type == FixtureType.DOOR_TOP_HINGE_SINGLE:
            if "stack" in fixture.name:  # wall stack cabinets not valid
                return False
            fxtr_bottom_z = fixture.pos[2] + fixture.bottom_offset[2]
            height_check = 1.0 <= fxtr_bottom_z <= 1.60
            if not height_check:
                return False
        return True
    elif fixture_type in [
        FixtureType.DOOR_HINGE_DOUBLE,
        FixtureType.DOOR_TOP_HINGE_DOUBLE,
    ]:
        cls_check = any([isinstance(fixture, cls) for cls in [HingeCabinet]])
        if not cls_check:
            return False
        if fixture_type == FixtureType.DOOR_TOP_HINGE_DOUBLE:
            if "stack" in fixture.name:  # wall stack cabinets not valid
                return False
            fxtr_bottom_z = fixture.pos[2] + fixture.bottom_offset[2]
            height_check = 1.0 <= fxtr_bottom_z <= 1.60
            if not height_check:
                return False
        return True
    elif fixture_type == FixtureType.TOASTER:
        return isinstance(fixture, Toaster)
    elif fixture_type == FixtureType.TOP_DRAWER:
        height_check = 0.7 <= fixture.pos[2] <= 0.9
        return height_check and isinstance(fixture, Drawer)
    elif fixture_type == FixtureType.STOOL:
        return isinstance(fixture, Stool)
    elif fixture_type == FixtureType.ISLAND:
        return isinstance(fixture, Counter) and "island" in fixture.name
    elif fixture_type == FixtureType.COUNTER_NON_CORNER:
        return isinstance(fixture, Counter) and "corner" not in fixture.name
    else:
        raise ValueError
