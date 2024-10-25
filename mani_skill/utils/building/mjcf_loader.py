from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, TypedDict

from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.utils.building.articulation_builder import ArticulationBuilder
from mani_skill.utils.structs import Actor, Articulation

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene

from ._mjcf_loader import MJCFLoader as SAPIENMJCFLoader


class ParsedMJCFData(TypedDict):
    articulation_builders: List[ArticulationBuilder]
    actor_builders: List[ActorBuilder]
    cameras: List[Any]


class MJCFLoader(SAPIENMJCFLoader):
    """
    Wrapper for the SAPIEN MJCF Loader to support easy parallelization
    """

    scene: ManiSkillScene
    name: str = None
    disable_self_collisions: bool = False

    def parse(self, mjcf_file, package_dir=None) -> ParsedMJCFData:
        articulation_builders, actor_builders, cameras = super().parse(
            mjcf_file, package_dir
        )
        for i, a in enumerate(articulation_builders):
            if len(articulation_builders) > 1:
                a.set_name(f"{self.name}-articulation-{i}")
            else:
                a.set_name(f"{self.name}")
            if self.disable_self_collisions:
                for l in a.link_builders:
                    # NOTE (stao): Currently this may not be working as intended
                    l.collision_groups[2] |= 1 << 29
        for i, b in enumerate(actor_builders):
            b.set_name(f"{self.name}-actor-{i}")
        return dict(
            articulation_builders=articulation_builders,
            actor_builders=actor_builders,
            cameras=cameras,
        )

    def load(
        self,
        mjcf_file: str,
        package_dir=None,
        name=None,
        scene_idxs=None,
    ) -> Articulation:
        """
        Args:
            urdf_file: filename for URDL file
            srdf_file: SRDF for urdf_file. If srdf_file is None, it defaults to the ".srdf" file with the same as the urdf file
            package_dir: base directory used to resolve asset files in the URDF file. If an asset path starts with "package://", "package://" is simply removed from the file name
            name (str): name of the created articulation
            scene_idxs (list[int]): the ids of the scenes to build the objects in
        Returns:
            returns a single Articulation loaded from the URDF file. It throws an error if multiple objects exists
        """
        if name is not None:
            self.name = name
        _parsed_mjcf_data = self.parse(mjcf_file, package_dir)
        articulation_builders = _parsed_mjcf_data["articulation_builders"]
        _parsed_mjcf_data["actor_builders"]
        cameras = _parsed_mjcf_data["cameras"]

        articulations: List[Articulation] = []
        for b in articulation_builders[:1]:
            b.set_scene_idxs(scene_idxs)
            b.disable_self_collisions = self.disable_self_collisions
            articulations.append(b.build())

        actors: List[Actor] = []
        # for b in actor_builders:
        #     actors.append(b.build())

        if len(cameras) > 0:
            name2entity = dict()
            for a in articulations:
                for sapien_articulation in a._objs:
                    for l in sapien_articulation.links:
                        name2entity[l.name] = l.entity

            for a in actors:
                name2entity[a.name] = a

            # TODO (stao): support extracting sensors
            # for scene_idx, scene in enumerate(self.scene.sub_scenes):
            #     for cam in cameras:
            #         cam_component = RenderCameraComponent(cam["width"], cam["height"])
            #         if cam["fovx"] is not None and cam["fovy"] is not None:
            #             cam_component.set_fovx(cam["fovx"], False)
            #             cam_component.set_fovy(cam["fovy"], False)
            #         elif cam["fovy"] is None:
            #             cam_component.set_fovx(cam["fovx"], True)
            #         elif cam["fovx"] is None:
            #             cam_component.set_fovy(cam["fovy"], True)

            #         cam_component.near = cam["near"]
            #         cam_component.far = cam["far"]
            #         name2entity[f"scene-{scene_idx}_{cam['reference']}"].add_component(
            #             cam_component
            #         )

        return articulations[0]
