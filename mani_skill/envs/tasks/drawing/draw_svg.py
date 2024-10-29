import math
import random
from typing import Dict




import mani_skill.envs.utils.randomization as randomization
import numpy as np
import sapien
import torch
from mani_skill.agents.robots.panda.panda_stick import PandaStick
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table.scene_builder import \
    TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SceneConfig, SimConfig
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import quat2mat

import svgpathtools
from svgpathtools import Line, QuadraticBezier, CubicBezier

@register_env("DrawSVG-v1", max_episode_steps=300)
class DrawSVG(BaseEnv):

    MAX_DOTS = 1000
    """
    The total "ink" available to use and draw with before you need to call env.reset. NOTE that on GPU simulation it is not recommended to have a very high value for this as it can slow down rendering
    when too many objects are being rendered in many scenes.
    """
    DOT_THICKNESS = 0.003
    """thickness of the paint drawn on to the canvas"""
    CANVAS_THICKNESS = 0.02
    """How thick the canvas on the table is"""
    BRUSH_RADIUS = 0.01
    """The brushes radius"""
    BRUSH_COLORS = [[0.8, 0.2, 0.2, 1]]
    """The colors of the brushes. If there is more than one color, each parallel environment will have a randomly sampled color."""
    THRESHOLD = 0.05

    SUPPORTED_REWARD_MODES = ["sparse"]

    SUPPORTED_ROBOTS: ["panda_stick"]  # type: ignore
    agent: PandaStick

    def __init__(self, *args, svg=None, robot_uids="panda_stick", **kwargs):
        if svg == None:
            self.svg = """M 100,30
                C 60,30 30,60 30,100
                C 30,140 60,170 100,190
                C 140,170 170,140 170,100
                C 170,60 140,30 100,30"""
        else:
            self.svg = svg

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        # we set contact_offset to a small value as we are not expecting to make any contacts really apart from the brush hitting the canvas too hard.
        # We set solver iterations very low as this environment is not doing a ton of manipulation (the brush is attached to the robot after all)
        return SimConfig(
            sim_freq=100,
            control_freq=20,
            scene_config=SceneConfig(
                contact_offset=0.01,
                solver_position_iterations=4,
                solver_velocity_iterations=0,
            ),
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.8], target=[0, 0, 0.1])
        return [
            CameraConfig(
                "base_camera",
                pose=pose,
                width=320,
                height=240,
                fov=1.2,
                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.8], target=[0, 0, 0.1])
        return CameraConfig(
            "render_camera",
            pose=pose,
            width=1280,
            height=960,
            fov=1.2,
            near=0.01,
            far=100,
        )

    def _load_scene(self, options: dict):

        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0)
        self.table_scene.build()
        


        def bezier_points(points, num_points=10):
            """
            Generate points along a quadratic or cubic Bezier curve using numpy.
            
            Args:
                *points: Variable number of control points as (x,y) tuples:
                        - For quadratic: (start, control, end)
                        - For cubic: (start, control1, control2, end)
                num_points: Number of points to generate along curve
                
            Returns:
                Array of points along the curve, shape (num_points, 2)
            """
            points = np.array(points)
            
            if len(points) not in [3, 4]:
                raise ValueError("Must provide either 3 points (quadratic) or 4 points (cubic)")
            t = np.linspace(0, 1, num_points).reshape(-1, 1)
            
            if len(points) == 3:
                p0, p1, p2 = points
                points = (
                    (1 - t)**2 * p0 +
                    2 * t * (1 - t) * p1 +
                    t**2 * p2
                )
            else:
                p0, p1, p2, p3 = points
                points = (
                    (1 - t)**3 * p0 +
                    3 * t * (1 - t)**2 * p1 +
                    3 * t**2 * (1 - t) * p2 +
                    t**3 * p3
                )
            
            return points

        parsed_svg = svgpathtools.parse_path(self.svg)
        
        if not parsed_svg.iscontinuous():
            raise ValueError("SVG path must be continuous")
        lines = []
        for path in parsed_svg:
            if isinstance(path,QuadraticBezier) or isinstance(path, CubicBezier):
                pts = bezier_points([[p.real, p.imag] for p in path.bpoints()])
                for i in range(len(pts)-1):
                    lines.append([pts[i],pts[i+1]])
            if isinstance(path, Line):
                lines.append([[p.real, p.imag] for p in path.bpoints()])
        lines = np.array(lines)
        lines = (lines / np.max(lines)) * 0.3
        lines = np.concatenate([lines, np.ones((*lines.shape[:-1],1)) * 0.02], -1)
        center = lines[:,:1,:].mean(axis=0) * np.array([[1,1,0]])
        lines = lines - center
        self.original_points = np.concatenate((lines[:1,0],lines[:,1,:]))

        def create_goal_outline(name="svg", base_color=None):
            midpoints = np.mean(lines, axis=1)
            box_half_ws = np.linalg.norm(lines[:,1]-lines[:,0], axis=1) / 2

            box_half_h = 0.01 / 2
            half_thickness = 0.001 / 2
            mids = midpoints[:,:2]
            ends = lines[:,1,:2] # 36, 2
            vec = ends-mids
            print(vec)
            angles = np.arccos(vec.dot(np.array([0,-1])) / (box_half_ws*2))

            builder = self.scene.create_actor_builder()
            for i,m in enumerate(midpoints):
                pose = sapien.Pose(p=m, q = euler2quat(0,0,angles[i]))
                
                builder.add_box_visual(
                    pose=pose,
                    half_size = [box_half_ws[i], box_half_h, half_thickness], # type: ignore
                    material=sapien.render.RenderMaterial(
                        base_color=base_color,
                    ),
                )
            

            return builder.build_kinematic(name=name)

        # build a white canvas on the table
        self.canvas = self.scene.create_actor_builder()
        self.canvas.add_box_visual(
            half_size=[0.4, 0.6, self.CANVAS_THICKNESS / 2],
            material=sapien.render.RenderMaterial(base_color=[1, 1, 1, 1]),
        )
        self.canvas.add_box_collision(
            half_size=[0.4, 0.6, self.CANVAS_THICKNESS / 2],
        )
        self.canvas.initial_pose = sapien.Pose(p=[-0.1, 0, self.CANVAS_THICKNESS / 2])
        self.canvas = self.canvas.build_static(name="canvas")

        self.dots = []
        self.dot_pos = None
        color_choices = torch.randint(0, len(self.BRUSH_COLORS), (self.num_envs,))
        for i in range(self.MAX_DOTS):
            actors = []
            if len(self.BRUSH_COLORS) > 1:
                for env_idx in range(self.num_envs):
                    builder = self.scene.create_actor_builder()
                    builder.add_cylinder_visual(
                        radius=self.BRUSH_RADIUS,
                        half_length=self.DOT_THICKNESS / 2,
                        material=sapien.render.RenderMaterial(
                            base_color=self.BRUSH_COLORS[color_choices[env_idx]]
                        ),
                    )
                    builder.set_scene_idxs([env_idx])
                    actor = builder.build_kinematic(name=f"dot_{i}_{env_idx}")
                    actors.append(actor)
                self.dots.append(Actor.merge(actors))
            else:
                builder = self.scene.create_actor_builder()
                builder.add_cylinder_visual(
                    radius=self.BRUSH_RADIUS,
                    half_length=self.DOT_THICKNESS / 2,
                    material=sapien.render.RenderMaterial(
                        base_color=self.BRUSH_COLORS[0]
                    ),
                )
                actor = builder.build_kinematic(name=f"dot_{i}")
                self.dots.append(actor)
        self.goal_tri = create_goal_outline(
            name="goal_tri",
            base_color=np.array([10, 10, 10,255]) / 255,
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.draw_step = 0
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            # target_pos = torch.zeros((b, 3))

            # target_pos[:, :2] = torch.rand((b, 2)) * 0.02 - 0.1
            # target_pos[:, -1] = 0.01
            # qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            # mats = quaternion_to_matrix(qs).to(self.device)
            # self.goal_tri.set_pose(Pose.create_from_pq(p=target_pos, q=qs))

            # self.vertices = torch.from_numpy(
            #     np.tile(self.original_verts, (b, 1, 1))
            # ).to(
            #     self.device
            # )  # b, 3, 3
            # self.vertices = (
            #     mats.double() @ self.vertices.transpose(-1, -2).double()
            # ).transpose(
            #     -1, -2
            # )  # apply rotation matrix
            # self.vertices += target_pos.unsqueeze(1)

            # self.triangles = self.generate_triangle_with_points(
            #     100, self.vertices[:, :, :-1]
            # )

            for dot in self.dots:
                # initially spawn dots in the table so they aren't seen
                dot.set_pose(
                    sapien.Pose(
                        p=[0, 0, -self.DOT_THICKNESS], q=euler2quat(0, np.pi / 2, 0)
                    )
                )

    def _after_control_step(self):
        if self.gpu_sim_enabled:
            self.scene._gpu_fetch_all()

        # This is the actual, GPU parallelized, drawing code.
        # This is not real drawing but seeks to mimic drawing by placing dots on the canvas whenever the robot is close enough to the canvas surface
        # We do not actually check if the robot contacts the table (although that is possible) and instead use a fast method to check.
        # We add a 0.005 meter of leeway to make it easier for the robot to get close to the canvas and start drawing instead of having to be super close to the table.
        robot_touching_table = (
            self.agent.tcp.pose.p[:, 2]
            < self.CANVAS_THICKNESS + self.DOT_THICKNESS + 0.005
        )
        robot_brush_pos = torch.zeros((self.num_envs, 3), device=self.device)
        robot_brush_pos[:, 2] = -self.DOT_THICKNESS
        robot_brush_pos[robot_touching_table, :2] = self.agent.tcp.pose.p[
            robot_touching_table, :2
        ]
        robot_brush_pos[robot_touching_table, 2] = (
            self.DOT_THICKNESS / 2 + self.CANVAS_THICKNESS
        )
        # move the next unused dot to the robot's brush position. All unused dots are initialized inside the table so they aren't visible
        new_dot_pos = Pose.create_from_pq(robot_brush_pos, euler2quat(0, np.pi / 2, 0))
        self.dots[self.draw_step].set_pose(new_dot_pos)
        if new_dot_pos.get_p()[:, -1] > 0:
            if self.dot_pos == None:
                self.dot_pos = new_dot_pos.get_p()[:, None, :]
            self.dot_pos = torch.cat(
                (self.dot_pos, new_dot_pos.get_p()[:, None, :]), dim=1
            )

        self.draw_step += 1

        # on GPU sim we have to call _gpu_apply_all() to apply the changes we make to object poses.
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()

    def evaluate(self):
        # out = self.success_check()
        return {"success": torch.zeros(self.num_envs).to(bool)}

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )

        # if "state" in self.obs_mode:
        #     obs.update(
        #         goal_pose = self.goal_tri.pose.raw_pose,
        #         tcp_to_verts_pos = self.vertices - self.agent.tcp.pose.p.unsqueeze(1),
        #         goal_pos=self.goal_tri.pose.p,
        #         vertices = self.vertices
        #     )

        return obs

    def generate_triangle_with_points(self, n, vertices):
        batch_size = vertices.shape[0]

        all_points = []

        for i in range(vertices.shape[1]):
            start_vertex = vertices[:, i, :]
            end_vertex = vertices[:, (i + 1) % vertices.shape[1], :]
            t = torch.linspace(0, 1, n + 2, device=vertices.device)[:-1]
            t = t.view(1, -1, 1).repeat(batch_size, 1, 2)
            intermediate_points = (
                start_vertex.unsqueeze(1) * (1 - t) + end_vertex.unsqueeze(1) * t
            )
            all_points.append(intermediate_points)
        all_points = torch.cat(all_points, dim=1)

        return all_points

    def success_check(self):
        if self.dot_pos == None or len(self.dot_pos) == 0:
            return torch.Tensor([False]).to(bool)
        drawn_pts = self.dot_pos[:, :, :-1]

        distance_matrix = torch.sqrt(
            torch.sum(
                (drawn_pts[:, :, None, :] - self.triangles[:, None, :, :]) ** 2, axis=-1
            )
        )

        Y_closeness = torch.min(distance_matrix, dim=1).values < self.THRESHOLD
        return torch.Tensor([torch.all(Y_closeness)]).to(bool)