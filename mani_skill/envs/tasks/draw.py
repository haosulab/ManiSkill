from typing import Dict

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

MAX_DOTS = 300
DOT_THICKNESS = 0.004


@register_env("DrawLetters-v1", max_episode_steps=MAX_DOTS)
class DrawLettersEnv(BaseEnv):

    SUPPORTED_REWARD_MODES = ["none"]

    def __init__(self, *args, robot_uids="panda_stick", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=100, control_freq=20)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()
        # cheap way to un-texture table
        for part in self.table_scene.table._objs:
            for triangle in (
                part.find_component_by_type(sapien.render.RenderBodyComponent)
                .render_shapes[0]
                .parts
            ):
                triangle.material.set_base_color(np.array([255, 255, 255, 255]) / 255)
                triangle.material.set_base_color_texture(None)
                triangle.material.set_normal_texture(None)
                triangle.material.set_emission_texture(None)
                triangle.material.set_transmission_texture(None)
                triangle.material.set_metallic_texture(None)
                triangle.material.set_roughness_texture(None)
        self.dots = []
        for i in range(MAX_DOTS):
            builder = self.scene.create_actor_builder()
            builder.add_cylinder_visual(
                radius=0.01,
                half_length=DOT_THICKNESS / 2,
                material=sapien.render.RenderMaterial(base_color=[0.8, 0.2, 0.2, 1]),
            )
            actor = builder.build_kinematic(name=f"dot_{i}")
            self.dots.append(actor)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # NOTE (stao): for simplicity this task cannot handle partial resets
        self.draw_step = 0
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            for dot in self.dots:
                # initially spawn dots in the table so they aren't seen
                dot.set_pose(
                    sapien.Pose(p=[0, 0, -DOT_THICKNESS], q=euler2quat(0, np.pi / 2, 0))
                )

    def _after_control_step(self):
        # return super()._after_control_step()
        # robot_brush_pos = self.agent.tcp.pose.p
        # only draw if the robot is close to the table
        robot_touching_table = self.agent.tcp.pose.p[:, 2] < 2
        robot_brush_pos = torch.zeros((self.num_envs, 3), device=self.device)
        robot_brush_pos[:, 2] = -DOT_THICKNESS
        # print(robot_touching_table)
        robot_brush_pos[robot_touching_table, :2] = self.agent.tcp.pose.p[
            robot_touching_table, :2
        ]
        robot_brush_pos[robot_touching_table, 2] = DOT_THICKNESS
        self.dots[self.draw_step].set_pose(
            Pose.create_from_pq(robot_brush_pos, euler2quat(0, np.pi / 2, 0))
        )
        self.draw_step += 1
        self.scene._gpu_apply_all()

    def evaluate(self):
        return {}

    def _get_obs_extra(self, info: Dict):
        return dict()
