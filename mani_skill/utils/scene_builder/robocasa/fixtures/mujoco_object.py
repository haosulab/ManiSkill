import numpy as np

from mani_skill import ASSET_DIR
from mani_skill.envs.scene import ManiSkillScene


class MujocoObject:
    """
    Mimics the MujocoXMLObject class in robosuite
    """

    def __init__(
        self, scene: ManiSkillScene, xml: str, name: str, pos: np.ndarray = None
    ):
        self.name = name
        self.pos = np.array([0, 0, 0])
        if pos is not None:
            self.pos = pos
        self.quat = np.array([1, 0, 0, 0])
        # load the mjcf file
        self.scene = scene
        self.loader = scene.create_mjcf_loader()
        self.loader.visual_groups = [
            1
        ]  # for robocasa, 1 is visualized, 0 is collisions
        orig_xml = xml
        xml = (
            ASSET_DIR
            / "scene_datasets/robocasa_dataset/assets"
            / orig_xml
            / "model.xml"
        )
        if not xml.exists():
            xml = ASSET_DIR / "scene_datasets/robocasa_dataset/assets" / orig_xml
            parsed = self.loader.parse(xml, package_dir=xml / "./")
        else:
            parsed = self.loader.parse(xml, package_dir=xml / "../")
        assert (
            len(parsed["articulation_builders"]) + len(parsed["actor_builders"]) == 1
        ), "exepect robocasa xmls to either have one actor or one articulation"
        if len(parsed["actor_builders"]) == 1:
            self.actor_builder = parsed["actor_builders"][0]
        else:
            self.articulation_builder = parsed["articulation_builders"][0]

    """Functions from RoboCasa MujocoXMLObject class"""

    def set_pos(self, pos):
        self.pos = pos.copy()

    def set_scale(self, scale):
        self.loader.scale = scale
        self._scale = np.array(scale)
        self.size = np.multiply(self.size, self._scale)
        # TODO (stao): is there a nicer way to move this scale code elsewhere.
        if hasattr(self, "articulation_builder"):
            for link in self.articulation_builder.link_builders:
                for visual in link.visual_records:
                    visual.scale = np.array(visual.scale) * scale
                for col in link.collision_records:
                    col.scale = np.array(col.scale) * scale
        elif hasattr(self, "actor_builder"):
            for visual in self.actor_builder.visual_records:
                visual.scale = np.array(visual.scale) * scale
            for col in self.actor_builder.collision_records:
                col.scale = np.array(col.scale) * scale
