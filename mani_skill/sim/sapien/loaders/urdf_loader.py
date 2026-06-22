from __future__ import annotations

from typing import Any, TypedDict

from sapien.render import RenderCameraComponent
from sapien.wrapper.urdf_loader import URDFLoader as OriginalSapienURDFLoader

from mani_skill.sim.loaders.urdf import BaseURDFLoader
from mani_skill.sim.sapien.builders.actor_builder import SapienActorBuilder
from mani_skill.sim.sapien.builders.articulation_builder import (
    SapienArticulationBuilder,
)
from mani_skill.sim.sapien.sim import SapienSim
from mani_skill.sim.sapien.structs.actor import SapienActor
from mani_skill.sim.sapien.structs.articulation import SapienArticulation


class ParsedURDFData(TypedDict):
    articulation_builders: list[SapienArticulationBuilder]
    actor_builders: list[SapienActorBuilder]
    cameras: list[Any]


class SapienURDFLoader(OriginalSapienURDFLoader, BaseURDFLoader):
    sim: SapienSim
    name: str = ""
    disable_self_collisions: bool = False
    fix_root_link: bool = True
    load_multiple_collisions_from_file: bool = False
    scale: float = 1.0

    @property
    def scene(self):
        return self.sim

    def parse(self, urdf_file, srdf_file=None, package_dir=None) -> ParsedURDFData:
        articulation_builders, actor_builders, cameras = super().parse(
            urdf_file, srdf_file, package_dir
        )
        for i, a in enumerate(articulation_builders):
            if len(articulation_builders) > 1:
                a.set_name(f"{self.name}-articulation-{i}")
            else:
                a.set_name(f"{self.name}")
            if self.disable_self_collisions:
                for link_builder in a.link_builders:
                    # NOTE (stao): Currently this may not be working as intended
                    link_builder.collision_groups[2] |= 1 << 29
        for i, b in enumerate(actor_builders):
            b.set_name(f"{self.name}-actor-{i}")
        return dict(
            articulation_builders=articulation_builders,
            actor_builders=actor_builders,
            cameras=cameras,
        )

    def load_file_as_articulation_builder(
        self, urdf_file, srdf_file=None, package_dir=None
    ) -> SapienArticulationBuilder:
        return super().load_file_as_articulation_builder(
            urdf_file, srdf_file, package_dir
        )

    def load(
        self,
        urdf_file: str,
        srdf_file=None,
        package_dir=None,
        name=None,
        scene_idxs=None,
    ) -> SapienArticulation:
        """
        Args:
            urdf_file: filename for URDL file
            srdf_file: SRDF for urdf_file. If srdf_file is None, it defaults to the ".srdf" file
                with the same as the urdf file
            package_dir: base directory used to resolve asset files in the URDF file.
            name (str): name of the created articulation
            scene_idxs (list[int]): the ids of the scenes to build the objects in
        Returns:
            returns a single Articulation loaded from the URDF file. It throws an error if multiple
            objects exists
        """
        if name is not None:
            self.name = name
        _parsed_urdf_data = self.parse(urdf_file, srdf_file, package_dir)
        articulation_builders = _parsed_urdf_data["articulation_builders"]
        actor_builders = _parsed_urdf_data["actor_builders"]
        cameras = _parsed_urdf_data["cameras"]

        if len(articulation_builders) > 1 or len(actor_builders) != 0:
            raise Exception(
                "URDF contains multiple objects, call load_multiple instead"
            )

        articulations: list[SapienArticulation] = []
        for b in articulation_builders:
            b.set_scene_idxs(scene_idxs)
            b.disable_self_collisions = self.disable_self_collisions
            articulations.append(b.build())

        actors: list[SapienActor] = []
        for b in actor_builders:
            actors.append(b.build())

        if len(cameras) > 0:
            name2entity = dict()
            for a in articulations:
                for sapien_articulation in a._objs:
                    for link in sapien_articulation.links:
                        name2entity[link.name] = link.entity

            for a in actors:
                name2entity[a.name] = a

            for scene_idx in range(self.sim.num_envs):
                for cam in cameras:
                    cam_component = RenderCameraComponent(cam["width"], cam["height"])
                    if cam["fovx"] is not None and cam["fovy"] is not None:
                        cam_component.set_fovx(cam["fovx"], False)
                        cam_component.set_fovy(cam["fovy"], False)
                    elif cam["fovy"] is None:
                        cam_component.set_fovx(cam["fovx"], True)
                    elif cam["fovx"] is None:
                        cam_component.set_fovy(cam["fovy"], True)

                    cam_component.near = cam["near"]
                    cam_component.far = cam["far"]
                    name2entity[f"scene-{scene_idx}_{cam['reference']}"].add_component(
                        cam_component
                    )

        return articulations[0]
