import logging
from typing import Any, Dict, Union

import numpy as np
import sapien
import sapien.render
import torch
from sapien.physx import PhysxMaterial

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

try:  # Optional dependency for convex decomposition
    import coacd  # noqa: F401

    _HAS_COACD = True
except ImportError:
    _HAS_COACD = False

logger = logging.getLogger(__name__)


@register_env("PickDishFromRack-v1", max_episode_steps=100)
class PickDishFromRackEnv(BaseEnv):
    """
    **Task Description:**
    Pick up the plate from the dish rack (where it starts vertically) and place it flat on the table.

    **Randomizations:**
    - The plate starts vertically in the dish rack.
    - The dish rack pose is randomized slightly on the tabletop.

    **Success Conditions:**
    - The plate is removed from the rack and placed flat on the table while the robot releases it.
    """

    agent: Union[Panda, Fetch]

    _rack_mesh_path = PACKAGE_ASSET_DIR / "dish_into_rack/dish_rack_with_connectors.stl"
    _plate_visual_mesh_path = (
        PACKAGE_ASSET_DIR / "dish_into_rack/white_ceramic_serving_bowl.glb"
    )
    _plate_mesh_source_radius = 0.5  # Radius of the raw OBJ (measured once offline)
    _plate_mesh_source_height = 0.2494586706161499  # OBJ height once flattened
    _plate_mesh_flat_quat = [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]  # Rotate mesh so Z is the plate normal
    _rack_scale = 0.0015  # Rack to match

    # Plate geometry parameters (meters) - SAME AS PLACE_DISH_IN_RACK
    _plate_outer_radius = 0.09  # Desired radius after scaling the OBJ
    _plate_inner_radius = 0.07  # Left for planners/controllers that rely on this value
    _plate_density = 300.0  # Lower density = lighter = easier to hold
    _plate_total_height = (
        _plate_mesh_source_height * (_plate_outer_radius / _plate_mesh_source_radius)
    )
    # Legacy attributes kept for compatibility with existing planners/utilities.
    _plate_base_thickness = _plate_total_height * 0.25
    _plate_rim_height = _plate_total_height - _plate_base_thickness
    _plate_extent = np.array(
        [_plate_outer_radius * 2, _plate_outer_radius * 2, _plate_total_height]
    )
    _plate_spawn_buffer = 0.002  # Small buffer to prevent initial interpenetration

    _rack_extent = np.array([0.12060600281, 0.16782440567, 0.085])  # Normal rack size
    _plate_goal_offset = np.array([0.0, 0.0, 0.15])  # Above rack slots (same as place task)
    _rack_position = np.array([-0.1, 0.1, 0])  # SAME AS PLACE - Rack position closer to robot workspace
    _plate_goal_position = np.array([-0.35, -0.15, 0])  # Reverse of place - where plate started in place task
    _plate_support_radius = 0.015
    _plate_support_height = 0.0  # No pedestal - plate flush with table

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**23, max_rigid_patch_count=2**17
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, -0.25, 0.35], target=[0.0, 0.0, 0.05])
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
        pose = sapien_utils.look_at([0.65, -0.35, 0.35], [0.05, 0.0, 0.1])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: Dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _get_obs_agent(self):
        obs = super()._get_obs_agent()
        for key in ("world__T__ee", "world__T__root"):
            value = obs.get(key, None)
            if isinstance(value, torch.Tensor) and value.ndim == 3:
                obs[key] = value.reshape(value.shape[0], -1)
        return obs

    def _load_scene(self, options: Dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.plate = self._build_plate()
        self.dish_rack = self._build_rack()
        self._plate_gravity_enabled = False

    def _build_plate(self):
        """Build the plate directly from the high-fidelity ceramic bowl mesh."""
        builder = self.scene.create_actor_builder()

        physical_material = PhysxMaterial(
            static_friction=20.0,
            dynamic_friction=20.0,
            restitution=0.0,
        )

        collision_scale = float(
            self._plate_outer_radius / self._plate_mesh_source_radius
        )
        mesh_pose = sapien.Pose(q=self._plate_mesh_flat_quat)

        if _HAS_COACD:
            builder.add_multiple_convex_collisions_from_file(
                filename=str(self._plate_visual_mesh_path),
                scale=[collision_scale, collision_scale, collision_scale],
                pose=mesh_pose,
                material=physical_material,
                density=self._plate_density,
                decomposition="coacd",
            )
        else:
            logger.warning(
                "coacd not installed; falling back to nonconvex collision for plate. "
                "Run `pip install coacd` for better plate contacts."
            )
            builder.add_nonconvex_collision_from_file(
                filename=str(self._plate_visual_mesh_path),
                scale=[collision_scale, collision_scale, collision_scale],
                pose=mesh_pose,
                material=physical_material,
                density=self._plate_density,
            )

        plate_visual_material = sapien.render.RenderMaterial(
            base_color=[1.0, 1.0, 1.0, 1.0],
            specular=0.4,
            roughness=0.2,
            metallic=0.0,
        )

        builder.add_visual_from_file(
            filename=str(self._plate_visual_mesh_path),
            scale=[collision_scale, collision_scale, collision_scale],
            pose=mesh_pose,
            material=plate_visual_material,
        )

        builder.initial_pose = sapien.Pose()
        return builder.build(name="plate")


    def _build_rack(self):
        builder = self.scene.create_actor_builder()

        # Collision geometry matching the exact STL mesh: base + 4 vertical dividers
        # Extracted from dish_rack_with_connectors.stl after 0.0015 scaling
        # (Same as place_dish_in_rack.py for consistency)

        # Base plate dimensions (extracted from STL)
        base_width = 0.180906  # X
        base_depth = 0.251737  # Y
        base_thickness = 0.009999  # Z (about 1cm)
        base_center = [0.004323, 0.007770, -0.057710]

        builder.add_box_collision(
            half_size=[base_width / 2, base_depth / 2, base_thickness / 2],
            pose=sapien.Pose(p=base_center)
        )

        # Vertical divider positions and heights (extracted from STL)
        rack_width = self._rack_extent[0]  # Width for dividers to span across
        divider_y_positions = [-0.105254, -0.046585, 0.015046, 0.074831]  # 4 dividers
        divider_heights = [0.125304, 0.122871, 0.122871, 0.125304]
        divider_z_centers = [-0.001098, -0.002315, -0.002315, -0.001098]
        divider_thickness = 0.003  # 3mm thick

        # Add vertical dividers on top of base
        # Dividers run in X direction (left-right) so plates slide in from the front (Y direction)
        for y_pos, height, z_center in zip(divider_y_positions, divider_heights, divider_z_centers):
            builder.add_box_collision(
                half_size=[rack_width / 2, divider_thickness, height / 2],
                pose=sapien.Pose(p=[0, y_pos, z_center])
            )

        # Keep the visual mesh (now centered at origin)
        builder.add_visual_from_file(
            filename=str(self._rack_mesh_path),
            scale=[self._rack_scale] * 3,
        )
        builder.initial_pose = sapien.Pose()
        return builder.build_kinematic(name="dish_rack")

    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        device = self.device
        with torch.device(device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Get table top Z coordinate
            table_p_arr = np.asarray(self.table_scene.table.pose.p).ravel()
            table_z = float(table_p_arr[-1])
            table_top_z = table_z + float(self.table_scene.table_height)

            # Position rack on table (SAME AS PLACE_DISH_IN_RACK)
            rack_pos = torch.zeros((b, 3), device=device)
            rack_pos[:] = torch.tensor(self._rack_position, device=device)
            rack_pos[:, 2] = table_top_z + float(self._rack_extent[2])  # Same as place task
            rack_pose = Pose.create_from_pq(p=rack_pos)
            self.dish_rack.set_pose(rack_pose)

            # Place plate VERTICALLY between the dividers in the rack
            plate_pos = rack_pos.clone()
            # X,Y same as rack center

            # When plate is vertical (after 90° rotation around X), its radius extends in Z direction
            # Bottom of plate = center_z - plate_outer_radius
            # Position center so bottom is at table level (bottom won't clip through)
            plate_pos[:, 2] = table_top_z + self._plate_outer_radius  # Center at radius height

            # Plate vertical - normal pointing in +X direction (perpendicular to dividers)
            # Dividers run in X (left-right), plate face should be perpendicular to Y (front-back)
            # Quaternion for 90deg rotation around X axis to make plate stand vertical
            vertical_quat = torch.tensor(
                [[0.7071068, 0.7071068, 0, 0]],  # 90 deg around X axis
                device=device,
                dtype=torch.float32,
            ).repeat(b, 1)

            plate_pose = Pose.create_from_pq(p=plate_pos, q=vertical_quat)
            self.plate.set_pose(plate_pose)

            # Keep the plate fixed in the rack until it is grasped
            self.plate.disable_gravity = True

            # Zero velocities so the plate remains stationary until grasped
            zero_velocity = torch.zeros((b, 3), device=device)
            self.plate.set_linear_velocity(zero_velocity)
            self.plate.set_angular_velocity(zero_velocity)
            self._plate_gravity_enabled = False

    def step(self, action):
        obs = super().step(action)
        if (
            not self._plate_gravity_enabled
            and bool(self.agent.is_grasping(self.plate).any())
        ):
            self.plate.disable_gravity = False
            self._plate_gravity_enabled = True
        return obs

    def evaluate(self):
        plate_pos = self.plate.pose.p
        goal_pos = torch.tensor(
            self._plate_goal_position, device=self.device, dtype=plate_pos.dtype
        ).unsqueeze(0).repeat(plate_pos.shape[0], 1)

        # Add Z component to goal (should be on table surface)
        table_p_arr = np.asarray(self.table_scene.table.pose.p).ravel()
        table_z = float(table_p_arr[-1])
        table_top_z = table_z + float(self.table_scene.table_height)
        goal_pos[:, 2] = table_top_z + self._plate_total_height / 2.0

        plate_to_goal = torch.linalg.norm(plate_pos - goal_pos, dim=1)

        # Check plate is horizontal (normal vector should point up)
        rot_mats = quaternion_to_matrix(self.plate.pose.q)
        plate_norm = rot_mats[..., 2]
        plate_horizontal = torch.abs(plate_norm[..., 2]) >= 0.85  # Normal points up/down

        is_grasped = self.agent.is_grasping(self.plate)
        is_static = self.plate.is_static(lin_thresh=0.02, ang_thresh=0.4)

        # Check that plate is above table surface
        above_table = plate_pos[:, 2] > table_top_z - 0.01

        # Plate should be far from rack (removed from rack)
        rack_pos = self.dish_rack.pose.p
        plate_to_rack = torch.linalg.norm(plate_pos - rack_pos, dim=1)
        away_from_rack = plate_to_rack >= 0.15

        close_to_goal = plate_to_goal <= 0.1
        success = close_to_goal & plate_horizontal & (~is_grasped) & is_static & above_table & away_from_rack

        return {
            "success": success,
            "plate_close_to_goal": close_to_goal,
            "plate_horizontal": plate_horizontal,
            "is_static": is_static,
            "is_grasped": is_grasped,
            "above_table": above_table,
            "away_from_rack": away_from_rack,
        }

    def _get_obs_extra(self, info: Dict):
        plate_pose = self.plate.pose
        rack_pose = self.dish_rack.pose
        obs = {
            "plate_pos": plate_pose.p,
            "plate_quat": plate_pose.q,
            "rack_pos": rack_pose.p,
            "rack_quat": rack_pose.q,
        }
        if "state" in self.obs_mode:
            goal_pos = torch.tensor(
                self._plate_goal_position, device=self.device, dtype=plate_pose.p.dtype
            ).unsqueeze(0).repeat(plate_pose.p.shape[0], 1)
            obs.update(
                plate_to_goal=plate_pose.p - goal_pos,
                tcp_to_plate=plate_pose.p - self.agent.tcp_pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict) -> torch.Tensor:
        """Compute dense reward for the task."""
        plate_pos = self.plate.pose.p
        goal_pos = torch.tensor(
            self._plate_goal_position, device=self.device, dtype=plate_pos.dtype
        ).unsqueeze(0).repeat(plate_pos.shape[0], 1)

        # Add Z component
        table_p_arr = np.asarray(self.table_scene.table.pose.p).ravel()
        table_z = float(table_p_arr[-1])
        table_top_z = table_z + float(self.table_scene.table_height)
        goal_pos[:, 2] = table_top_z + self._plate_total_height / 2.0

        plate_to_goal_dist = torch.linalg.norm(plate_pos - goal_pos, dim=1)

        # Distance reward
        reaching_reward = 1.0 - torch.tanh(5.0 * plate_to_goal_dist)

        # Orientation reward (plate should be horizontal)
        rot_mats = quaternion_to_matrix(self.plate.pose.q)
        plate_norm = rot_mats[..., 2]
        horizontal_alignment = torch.abs(plate_norm[..., 2])
        orientation_reward = horizontal_alignment  # Reward for being horizontal

        # Gripper release reward
        is_grasped = self.agent.is_grasping(self.plate)
        close_to_goal = plate_to_goal_dist <= 0.1
        release_reward = torch.where(
            close_to_goal,
            torch.where(is_grasped, torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device)),
            torch.tensor(0.0, device=self.device)
        )

        # Success bonus
        success = info["success"].float()
        success_reward = success * 5.0

        reward = reaching_reward + orientation_reward + release_reward + success_reward
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict) -> torch.Tensor:
        """Compute normalized dense reward."""
        return self.compute_dense_reward(obs, action, info) / 8.0
