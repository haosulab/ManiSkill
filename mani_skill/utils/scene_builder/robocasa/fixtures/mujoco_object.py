from typing import Optional

import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.scene_builder.robocasa.utils.scene_utils import ROBOCASA_ASSET_DIR


def site_pos(elem):
    if "pos" in elem.keys():
        out = np.fromstring(elem.attrib["pos"], sep=" ", dtype=np.float32)
    else:
        out = np.array([0, 0, 0], dtype=np.float32)
    return out


class MujocoObject:
    """
    Mimics the MujocoXMLObject class in robosuite
    """

    def __init__(
        self,
        scene: ManiSkillScene,
        xml: Optional[str],
        name: str,
        pos: np.ndarray = None,
        scale: float = 1,
    ):
        self.name = name
        self.pos = np.array([0, 0, 0])
        if pos is not None:
            self.pos = np.array(pos)
        self.quat = np.array([1, 0, 0, 0])
        # load the mjcf file
        self.scene = scene
        self.loader = scene.create_mjcf_loader()
        self.loader.visual_groups = [
            1
        ]  # for robocasa, 1 is visualized, 0 is collisions
        if xml is not None:
            orig_xml = xml
            xml = ROBOCASA_ASSET_DIR / orig_xml / "model.xml"
            if not xml.exists():
                xml = ROBOCASA_ASSET_DIR / orig_xml
                parsed = self.loader.parse(xml, package_dir=xml / "./")
            else:
                parsed = self.loader.parse(xml, package_dir=xml / "../")
            assert (
                len(parsed["articulation_builders"]) + len(parsed["actor_builders"])
                == 1
            ), "exepect robocasa xmls to either have one actor or one articulation"
            if len(parsed["actor_builders"]) == 1:
                self.actor_builder = parsed["actor_builders"][0]
            else:
                self.articulation_builder = parsed["articulation_builders"][0]

        # set up exterior and interior sites
        self._bounds_sites = dict()
        for postfix in [
            "ext_p0",
            "ext_px",
            "ext_py",
            "ext_pz",
            "int_p0",
            "int_px",
            "int_py",
            "int_pz",
        ]:
            for elem in self.loader.xml.findall(".//*site"):
                if elem.get("name") == f"{self.naming_prefix}{postfix}":
                    self._bounds_sites[postfix] = site_pos(elem)
                    break
            # site = find_elements(
            #     self.worldbody,
            #     tags="site",
            #     attribs={"name": "{}{}".format(self.naming_prefix, postfix)},
            #     return_first=True,
            # )
            # if site is None:
            #     continue
            # rgba = string_to_array(site.get("rgba"))
            # if macros.SHOW_SITES:
            #     rgba[-1] = 1.0
            # else:
            #     rgba[-1] = 0.0
            # site.set("rgba", array_to_string(rgba))
            # self._bounds_sites[postfix] = site
        if scale != 1:
            # NOTE (stao): this is a hacky way to try and imitate the original robocasa/robosuite scaling code.
            self.set_scale(scale)

    """Functions from RoboCasa MujocoXMLObject class"""

    def set_pos(self, pos):
        self.pos = np.array(pos)

    def set_euler(self, euler):
        self.quat = euler2quat(*euler)

    def set_scale(self, scale):
        """Based on https://github.com/ARISE-Initiative/robosuite/blob/robocasa_v0.1/robosuite/models/objects/objects.py#L507.

        scale is a float or a array with 3 floats
        """
        self.loader.scale = scale
        self._scale = np.array(scale)
        if hasattr(self, "size"):
            self.size = np.multiply(self.size, self._scale)
        # TODO (stao): is there a nicer way to move this scale code elsewhere.
        if hasattr(self, "articulation_builder"):
            for link in self.articulation_builder.link_builders:
                for visual in link.visual_records:
                    visual.pose = sapien.Pose(
                        p=np.multiply(visual.pose.p, scale), q=visual.pose.q
                    )
                    visual.scale = np.array(visual.scale) * scale
                for col in link.collision_records:
                    col.pose = sapien.Pose(
                        p=np.multiply(col.pose.p, scale), q=col.pose.q
                    )
                    if col.type == "cylinder" or col.type == "capsule":
                        col.radius *= scale[0]
                        col.length *= scale[1]
                    else:
                        col.scale = np.array(col.scale) * scale
                link.joint_record.pose_in_parent = sapien.Pose(
                    p=np.multiply(link.joint_record.pose_in_parent.p, scale),
                    q=link.joint_record.pose_in_parent.q,
                )
                link.joint_record.pose_in_child = sapien.Pose(
                    p=np.multiply(link.joint_record.pose_in_child.p, scale),
                    q=link.joint_record.pose_in_child.q,
                )
        elif hasattr(self, "actor_builder"):
            for visual in self.actor_builder.visual_records:
                visual.pose = sapien.Pose(
                    p=np.multiply(visual.pose.p, scale), q=visual.pose.q
                )
                if visual.type == "cylinder" or visual.type == "capsule":
                    visual.radius *= scale[0]
                    visual.length *= scale[1]
                else:
                    visual.scale = np.array(visual.scale) * scale
            for col in self.actor_builder.collision_records:
                col.pose = sapien.Pose(p=np.multiply(col.pose.p, scale), q=col.pose.q)
                if col.type == "cylinder" or col.type == "capsule":
                    col.radius *= scale[0]
                    col.length *= scale[1]
                else:
                    col.scale = np.array(col.scale) * scale
        if hasattr(self, "_bounds_sites"):
            # scale sites
            for k, v in self._bounds_sites.items():
                self._bounds_sites[k] = v * scale
            # for (_, elem) in site_pairs:
            #     s_pos = elem.get("pos")
            #     if s_pos is not None:
            #         s_pos = string_to_array(s_pos) * self._scale
            #         elem.set("pos", array_to_string(s_pos))

            #     s_size = elem.get("size")
            #     if s_size is not None:
            #         s_size_np = string_to_array(s_size)
            #         # handle cases where size is not 3 dimensional
            #         if len(s_size_np) == 3:
            #             s_size_np = s_size_np * self._scale
            #         elif len(s_size_np) == 2:
            #             scale = np.array(self._scale).reshape(-1)
            #             if len(scale) == 1:
            #                 s_size_np *= scale
            #             elif len(scale) == 3:
            #                 s_size_np[0] *= np.mean(scale[:2])  # width
            #                 s_size_np[1] *= scale[2]  # height
            #             else:
            #                 raise ValueError
            #         elif len(s_size_np) == 1:
            #             s_size_np *= np.mean(self._scale)
            #         else:
            #             raise ValueError
            #         s_size = array_to_string(s_size_np)
            #         elem.set("size", s_size)