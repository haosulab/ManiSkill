from collections import OrderedDict

import numpy as np
import sapien.core as sapien
from sapien.core import Pose

from mani_skill2 import ASSET_DIR
from mani_skill2.agents.configs.panda.defaults import PandaRealSensed435Config
from mani_skill2.agents.robots.panda import Panda
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.registration import register_gym_env
from mani_skill2.utils.sapien_utils import (
    get_articulation_max_impulse_norm,
    get_entity_by_name,
    look_at,
    set_articulation_render_material,
    vectorize_pose,
)


class AvoidObstaclesBaseEnv(BaseEnv):
    DEFAULT_EPISODE_JSON: str

    tcp: sapien.Link  # Tool Center Point of the robot

    def __init__(self, json_path=None, **kwargs):
        if json_path is None:
            json_path = self.DEFAULT_EPISODE_JSON
        self.episodes = load_json(json_path)
        self.episode_idx = None
        self.episode_config = None
        super().__init__(**kwargs)

    def _get_default_scene_config(self):
        scene_config = super()._get_default_scene_config()
        scene_config.contact_offset = 0.01
        return scene_config

    def reset(self, *args, seed=None, episode_idx=None, reconfigure=False, **kwargs):
        self.set_episode_rng(seed)
        if episode_idx is None:
            episode_idx = self._episode_rng.choice(len(self.episodes))
        if episode_idx != self.episode_idx:
            reconfigure = True
        self.episode_idx = episode_idx
        self.episode_config = self.episodes[episode_idx]
        return super().reset(
            *args, seed=self._episode_seed, reconfigure=reconfigure, **kwargs
        )

    def _build_cube(
        self,
        half_size,
        color=(1, 0, 0),
        name="cube",
        static=True,
        render_material: sapien.RenderMaterial = None,
    ):
        if render_material is None:
            render_material = self._renderer.create_material()
            render_material.set_base_color(np.hstack([color, 1.0]))

        builder = self._scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(half_size=half_size, material=render_material)
        if static:
            return builder.build_static(name)
        else:
            return builder.build(name)

    def _build_coord_frame_site(self, scale=0.1, name="coord_frame"):
        builder = self._scene.create_actor_builder()
        radius = scale * 0.05
        half_length = scale * 0.5
        builder.add_capsule_visual(
            sapien.Pose(p=[scale * 0.5, 0, 0], q=[1, 0, 0, 0]),
            radius=radius,
            half_length=half_length,
            color=[1, 0, 0],
            name="x",
        )
        builder.add_capsule_visual(
            sapien.Pose(p=[0, scale * 0.5, 0], q=[0.707, 0, 0, 0.707]),
            radius=radius,
            half_length=half_length,
            color=[0, 1, 0],
            name="y",
        )
        builder.add_capsule_visual(
            sapien.Pose(p=[0, 0, scale * 0.5], q=[0.707, 0, -0.707, 0]),
            radius=radius,
            half_length=half_length,
            color=[0, 0, 1],
            name="z",
        )
        actor = builder.build_static(name)
        # NOTE(jigu): Must hide upon creation to avoid pollute observations!
        actor.hide_visual()
        return actor

    def _load_actors(self):
        self._add_ground()

        # Add a wall
        if "wall" in self.episode_config:
            cfg = self.episode_config["wall"]
            self.wall = self._build_cube(cfg["half_size"], color=(1, 1, 1), name="wall")
            self.wall.set_pose(Pose(cfg["pose"][:3], cfg["pose"][3:]))

        self.obstacles = []
        for i, cfg in enumerate(self.episode_config["obstacles"]):
            actor = self._build_cube(
                cfg["half_size"], self._episode_rng.rand(3), name=f"obstacle_{i}"
            )
            actor.set_pose(Pose(cfg["pose"][:3], cfg["pose"][3:]))
            self.obstacles.append(actor)

        self.goal_site = self._build_coord_frame_site(scale=0.05)

    def _initialize_agent(self):
        qpos = self.episode_config["start_qpos"]
        # qpos = self.episode_config["end_qpos"]
        self.agent.reset(qpos)
        self.agent.robot.set_pose(Pose([0, 0, 0]))

    def _update_goal_to_obstacle_dist(self):
        obstacle_pos = [actor.pose.p for actor in self.obstacles]
        goal_pos = self.goal_pose.p
        goal_to_obstacle_dist = [np.linalg.norm(goal_pos - x) for x in obstacle_pos]
        self.goal_to_obstacle_dist = np.sort(goal_to_obstacle_dist)

    def _initialize_task(self):
        end_pose = self.episode_config["end_pose"]
        self.goal_pose = Pose(end_pose[:3], end_pose[3:])
        self.goal_site.set_pose(self.goal_pose)
        self._update_goal_to_obstacle_dist()

    def _get_obs_agent(self):
        obs = self.agent.get_proprioception()
        obs["base_pose"] = vectorize_pose(self.agent.robot.pose)
        return obs

    def _get_obs_extra(self) -> OrderedDict:
        tcp_pose = self.tcp.pose
        goal_pose = self.goal_pose
        return OrderedDict(
            tcp_pose=vectorize_pose(tcp_pose),
            goal_pose=vectorize_pose(goal_pose),
        )

    def evaluate(self, **kwargs) -> dict:
        tcp_pose_at_goal = self.goal_pose.inv() * self.tcp.pose
        pos_dist = np.linalg.norm(tcp_pose_at_goal.p)
        ang_dist = np.arccos(tcp_pose_at_goal.q[0]) * 2
        if ang_dist > np.pi:  # [0, 2 * pi] -> [-pi, pi]
            ang_dist = ang_dist - 2 * np.pi
        ang_dist = np.abs(ang_dist)
        ang_dist = np.rad2deg(ang_dist)
        success = pos_dist <= 0.025 and ang_dist <= 15
        return dict(pos_dist=pos_dist, ang_dist=ang_dist, success=success)

    def compute_dense_reward(self, info, **kwargs):
        if info["success"]:
            return 10.0

        pos_threshold = 0.025
        ang_threshold = 15
        reward = 0.0
        pos_dist, ang_dist = info["pos_dist"], info["ang_dist"]
        num_obstacles = len(self.obstacles)

        close_to_goal_reward = (
            4.0 * np.sum(pos_dist < self.goal_to_obstacle_dist) / num_obstacles
        )
        # close_to_goal_reward += 1 - np.tanh(pos_dist)
        angular_reward = 0.0

        smallest_g2o_dist = self.goal_to_obstacle_dist[0]
        if pos_dist < smallest_g2o_dist:
            angular_reward = 3.0 * (
                1 - np.tanh(np.maximum(ang_dist - ang_threshold, 0.0) / 180)
            )
            if ang_dist <= 25:
                close_to_goal_reward += 2.0 * (
                    1
                    - np.tanh(
                        np.maximum(pos_dist - pos_threshold, 0.0) / smallest_g2o_dist
                    )
                )

        contacts = self._scene.get_contacts()
        max_impulse_norm = get_articulation_max_impulse_norm(contacts, self.agent.robot)
        reward = close_to_goal_reward + angular_reward - 50.0 * max_impulse_norm
        return reward

    def _setup_cameras(self):
        self.render_camera = self._scene.add_camera(
            "render_camera", 512, 512, 1, 0.01, 10
        )
        self.render_camera.set_local_pose(look_at([1.5, 0, 1.5], [0.0, 0.0, 0.5]))

        base_camera = self._scene.add_camera(
            "base_camera", 128, 128, np.pi / 2, 0.01, 10
        )
        base_camera.set_local_pose(look_at([-0.25, 0, 1.2], [0.6, 0, 0.6]))
        self._cameras["base_camera"] = base_camera

    def _setup_viewer(self):
        super()._setup_viewer()
        self._viewer.set_camera_xyz(1.5, 0.0, 1.5)
        self._viewer.set_camera_rpy(0, -0.6, 3.14)

    def render(self, mode="human"):
        if mode in ["human", "rgb_array"]:
            self.goal_site.unhide_visual()
            ret = super().render(mode=mode)
            self.goal_site.hide_visual()
        else:
            ret = super().render(mode=mode)
        return ret


@register_gym_env("PandaAvoidObstacles-v0", max_episode_steps=500)
class PandaAvoidObstaclesEnv(AvoidObstaclesBaseEnv):
    DEFAULT_EPISODE_JSON = str(ASSET_DIR / "avoid_obstacles/panda_train_2k.json.gz")

    def _load_agent(self):
        self.robot_uuid = "panda"
        agent_config = PandaRealSensed435Config()
        self.agent = Panda(
            self._scene, self._control_freq, self._control_mode, config=agent_config
        )
        self.tcp: sapien.Link = get_entity_by_name(
            self.agent.robot.get_links(), self.agent._config.ee_link_name
        )
        set_articulation_render_material(self.agent.robot, specular=0.9, roughness=0.3)
