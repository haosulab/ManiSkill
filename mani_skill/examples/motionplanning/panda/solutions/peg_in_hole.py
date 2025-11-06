# peg_in_hole.py
import numpy as np
import torch
import sapien
import sapien.render as render
from typing import Dict, Any, Union

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils import sapien_utils
from mani_skill.agents.robots import Panda, Fetch


@register_env("PegInHole-v1", max_episode_steps=100000)
class PegInHoleEnv(BaseEnv):
    """Cylindrical peg on a table; insert into a socket with a small lead-in."""

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    # --- Geometry (meters)
    peg_radius = 0.015
    peg_length = 0.10

    socket_clearance = 0.003   # radial clearance at bottom guide ring
    funnel_extra     = 0.003   # extra clearance at top ring (lead-in / funnel)
    wall_thickness   = 0.010
    wall_height      = 0.050
    top_ring_height  = 0.015   # small funnel ring height

    # --- Success thresholds
    xy_tol         = 0.006
    depth_success  = 0.050

    # --- Placement
    socket_center_xy = np.array([0.25, 0.0], dtype=np.float32)

    @property
    def _default_human_render_camera_configs(self):
        from mani_skill.sensors.camera import CameraConfig
        pose = sapien_utils.look_at(eye=[0.70, 0.60, 0.60], target=[0.0, 0.0, 0.25])
        return [CameraConfig("viewer_cam", pose=pose, width=960, height=720, fov=np.pi/3, near=0.01, far=100.0)]

    def __init__(self, *args, robot_uids="panda", num_envs=1, reconfiguration_freq=None, **kwargs):
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0
        super().__init__(*args, robot_uids=robot_uids, reconfiguration_freq=reconfiguration_freq, num_envs=num_envs, **kwargs)

    # -------- Materials (RenderMaterial; no engine needed) --------
    def _mat(self, rgba):
        r, g, b, a = [float(x) for x in rgba]
        return render.RenderMaterial(base_color=[r, g, b, a])

    def _load_agent(self, options: dict):
        # Spawn above to avoid collisions on reconfigure
        super()._load_agent(options, sapien.Pose(p=[0, 0, 1.0]))

    def _load_scene(self, options: dict):
        # Table & ground
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()

        # Visual materials
        self.mat_grey   = self._mat([0.75, 0.75, 0.80, 1.0])
        self.mat_blue   = self._mat([0.20, 0.45, 0.95, 1.0])
        self.mat_orange = self._mat([0.95, 0.60, 0.25, 1.0])

        # Build socket & peg
        self.socket = self._build_socket_actor(self.socket_center_xy)
        self.peg    = self._build_peg_actor()

        # Optional overhead camera pose (debug)
        self._base_camera_pose = sapien_utils.look_at(eye=[0.60, 0.00, 0.80], target=[0.00, 0.00, 0.40])

    # ---- Builders -------------------------------------------------
    def _build_socket_actor(self, socket_xy: np.ndarray):
        """
        Two stacked square rings:
          - Top ring (lead-in): slightly larger
          - Bottom ring (guide): nominal clearance
        """
        R_bottom = self.peg_radius + self.socket_clearance
        R_top    = R_bottom + self.funnel_extra
        t        = self.wall_thickness
        h_top    = self.top_ring_height
        h_bot    = max(1e-3, self.wall_height - h_top)  # avoid 0
        h2_top   = 0.5 * h_top
        h2_bot   = 0.5 * h_bot

        def add_ring(builder, R, h2, z0):
            def wall(center_local, half_size):
                pose = sapien.Pose(p=center_local)
                builder.add_box_collision(pose=pose, half_size=half_size)
                builder.add_box_visual(pose=pose, half_size=half_size, material=self.mat_grey)
            # left/right (long in Y)
            wall([-(R + t/2), 0.0, z0 + h2], [t/2, (R + t), h2])
            wall([ +(R + t/2), 0.0, z0 + h2], [t/2, (R + t), h2])
            # front/back (long in X)
            wall([0.0,  +(R + t/2), z0 + h2], [(R + t), t/2, h2])
            wall([0.0,  -(R + t/2), z0 + h2], [(R + t), t/2, h2])

        b = self.scene.create_actor_builder()
        add_ring(b, R_top,    h2_top, 0.00)
        add_ring(b, R_bottom, h2_bot, h_top)
        b.initial_pose = sapien.Pose(p=[float(socket_xy[0]), float(socket_xy[1]), h2_top])
        return b.build_static(name="socket")

    def _build_peg_actor(self):
        """
        Plain cylindrical peg: capsule (axis is X in SAPIEN).
        We rotate to +Z at reset so the peg stands vertically.
        """
        b = self.scene.create_actor_builder()

        # Collision (heavier density helps resist sliding during grasp)
        b.add_capsule_collision(
            radius=self.peg_radius,
            half_length=self.peg_length / 2,
            density=5000.0,               # <- heavier = more stable against sliding
            patch_radius=0.01,            # safe no-op if binding ignores; improves contacts if present
            min_patch_radius=0.005
        )
        # Visual
        b.add_capsule_visual(
            radius=self.peg_radius,
            half_length=self.peg_length / 2,
            material=self.mat_orange
        )
        return b.build(name="peg")

    # ---- Episode init ---------------------------------------------
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Place socket on table (z ~= ring half-height)
            sock_xy = torch.tensor(self.socket_center_xy, dtype=torch.float32).repeat(b, 1)
            sock_z  = torch.full((b, 1), 0.5 * self.top_ring_height)
            self.socket.set_pose(Pose.create_from_pq(
                p=torch.cat([sock_xy, sock_z], dim=-1), q=[1, 0, 0, 0]
            ))

            # Peg: vertical, on table near origin (no stand, no pads)
            peg_xy = (torch.rand((b, 2)) - 0.5) * 0.12   # +/- 6 cm
            peg_z  = torch.full((b, 1), self.peg_length/2 + 0.002)   # slight lift to avoid interpenetration
            peg_p  = torch.cat([peg_xy, peg_z], dim=-1)

            # Rotate capsule axis X -> +Z (90° about +Y). Quaternion order is wxyz. :contentReference[oaicite:0]{index=0}
            q = torch.tensor([np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0], dtype=torch.float32)
            self.peg.set_pose(Pose.create_from_pq(p=peg_p, q=q))

    # ---- Success / observations -----------------------------------
    def evaluate(self):
        # success if peg tip is inside hole and sufficiently deep
        peg_p = self.peg.pose.p
        tip = peg_p - torch.tensor([0.0, 0.0, self.peg_length/2], device=peg_p.device)
        cx, cy = self.socket_center_xy
        center = torch.tensor([cx, cy], device=peg_p.device)
        xy_ok = torch.linalg.norm(tip[..., :2] - center, dim=-1) < self.xy_tol

        # depth relative to top of combined rings
        top_z = self.top_ring_height + self.wall_height
        deep_enough = (top_z - tip[..., 2]) > self.depth_success
        return {"success": xy_ok & deep_enough}

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                peg_pose=self.peg.pose.raw_pose,
                socket_pose=self.socket.pose.raw_pose,
            )
        return obs

    def compute_normalized_dense_reward(self, obs: Any, action: np.ndarray, info: Dict):
        # simple dense shaping toward socket center + entry plane
        peg_xy = self.peg.pose.p[..., :2]
        cx, cy = self.socket_center_xy
        d_xy = torch.linalg.norm(peg_xy - torch.tensor([cx, cy], device=peg_xy.device), dim=-1)
        tip_z = self.peg.pose.p[..., 2] - self.peg_length/2
        target_z = self.top_ring_height              # entry plane
        d_z = torch.clamp(target_z - tip_z, min=0.0)
        reward = -3.0 * d_xy - 1.0 * d_z + 1.0 * (d_xy < 0.015)
        return torch.clamp(reward / 5.0, -1.0, 1.0)
