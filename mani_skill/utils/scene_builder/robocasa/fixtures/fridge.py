from mani_skill.utils.scene_builder.robocasa.fixtures.fixture import Fixture


class Fridge(Fixture):
    """
    Fridge fixture class
    """

    def __init__(
        self,
        xml="fixtures/appliances/fridges/pack_1/model.xml",
        name="fridge",
        *args,
        **kwargs
    ):
        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )

    @property
    def nat_lang(self):
        return "fridge"
