from typing import List

import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat

from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder import SceneBuilder


class PlanarSceneBuilder(SceneBuilder):
    def build(self, build_config_idxs: List[int] = None):
        # ground - a strip with length along +x
        self.ground = build_ground(
            self.scene,
            floor_width=2,
            floor_length=100,
            altitude=0,
            xy_origin=(50 - 2, 0),
        )

        # background visual wall
        self.wall = self.scene.create_actor_builder()
        self.wall.add_box_visual(
            half_size=(1e-3, 65, 10),
            pose=sapien.Pose(p=[(50 - 2), 2, 0], q=euler2quat(0, 0, np.pi / 2)),
            material=sapien.render.RenderMaterial(
                base_color=np.array([0.3, 0.3, 0.3, 1])
            ),
        )
        self.wall.build_static(name="wall")
        self.scene_objects: List[sapien.Entity] = [self.ground, self.wall]

    def initialize(self, env_idx: torch.Tensor):
        b = len(env_idx)
        robot = self.env.agent.uid
        if robot == "hopper" or robot == "walker":
            # qpos sampled same as dm_control, but ensure no self intersection explicitly here
            random_qpos = torch.rand(b, self.env.agent.robot.dof[0])
            q_lims = self.env.agent.robot.get_qlimits()
            q_ranges = q_lims[..., 1] - q_lims[..., 0]
            random_qpos *= q_ranges
            random_qpos += q_lims[..., 0]

            # overwrite planar joint qpos - these are special for planar robots
            # first two joints are dummy rootx and rootz
            random_qpos[:, :2] = 0
            # y is axis of rotation of our planar robot (xz plane), so we randomize around it
            random_qpos[:, 2] = torch.pi * (2 * torch.rand(b) - 1)  # (-pi,pi)
            self.env.agent.reset(random_qpos)
        elif robot == "cheetah":
            raise NotImplementedError()
        else:
            raise NotImplementedError()
