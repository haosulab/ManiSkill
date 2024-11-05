from mani_skill.utils.scene_builder.robocasa.fixtures.fixture import Fixture


class Dishwasher(Fixture):
    """
    Dishwasher fixture class
    """

    def __init__(
        self,
        scene,
        xml="fixtures/appliances/dishwashers/pack_1/model.xml",
        name="dishwasher",
        *args,
        **kwargs
    ):
        super().__init__(
            scene, xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )

    @property
    def nat_lang(self):
        return "dishwasher"
