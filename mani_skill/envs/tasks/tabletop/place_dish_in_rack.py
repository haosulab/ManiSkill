import logging
from typing import Any, Dict, Union

import numpy as np
import sapien
import sapien.render
import torch

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
from sapien.physx import PhysxMaterial
from mani_skill.agents.controllers import PDEEPoseControllerConfig

try:  # Optional dependency for convex decomposition
    import coacd  # noqa: F401

    _HAS_COACD = True
except ImportError:
    _HAS_COACD = False

logger = logging.getLogger(__name__)


@register_env("PlaceDishInRack-v1", max_episode_steps=100)
class PlaceDishInRackEnv(BaseEnv):
    """
    **Task Description:**
    Pick up the plate and place it vertically into the upright slots of the dish rack.

    **Randomizations:**
    - The plate starts randomly on the table near the robot.
    - The dish rack pose is randomized slightly on the tabletop.

    **Success Conditions:**
    - The plate is upright and centered inside the dish rack while the robot releases it.
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

    # Plate geometry parameters (meters)
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
    # STL is now centered at origin, no offset needed

    _plate_goal_offset = np.array([0.0, 0.0, 0.15])  # Above rack slots
    _rack_position = np.array([-0.1, 0.1, 0])  # Rack position closer to robot workspace
    _plate_support_radius = 0.015
    _plate_support_height = 0.0  # No pedestal - plate flush with table

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.0, **kwargs):
        # Set noise to 0 to prevent joints from hitting limits
        self.robot_init_qpos_noise = 0.0
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=200,  # Moderate increase for better physics (default is 100)
            control_freq=20,  # Keep control frequency the same
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
        # Keep robot at normal position
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

        # Keep table at default position
        self.table_height_offset = 0.0

        self.plate = self._build_plate()
        # Add heavy damping so the plate stays still on the table until the robot touches it.
        # Without this, tiny numerical vibrations from the robot settling would slowly tip it upright.
        self.plate.set_linear_damping(5.0)
        self.plate.set_angular_damping(8.0)
        self.dish_rack = self._build_rack()
        self.plate_support = self._build_plate_support()

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

    def _build_plate_support(self):
        if self._plate_support_height <= 0.0:
            return None
        builder = self.scene.create_actor_builder()
        builder.add_cylinder_collision(
            radius=self._plate_support_radius,
            half_length=self._plate_support_height / 2,
        )
        builder.add_cylinder_visual(
            radius=self._plate_support_radius,
            half_length=self._plate_support_height / 2,
            material=sapien.render.RenderMaterial(base_color=[0.8, 0.8, 0.8, 1.0]),
        )
        builder.initial_pose = sapien.Pose()
        support = builder.build_kinematic(name="plate_support")
        return support

    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        device = self.device
        with torch.device(device):
            b = len(env_idx)
            # Initialize with arm pointing more downward and forward
            # This pose naturally brings the gripper lower and closer to the table
            better_qpos = np.array([
                0.0,           # joint1: neutral
                0.7,           # joint2: tilt down more (positive tilts forward/down)
                0.0,           # joint3: neutral
                -2.0,          # joint4: elbow bent (well within [-3.072, -0.070])
                0.0,           # joint5: neutral
                1.8,           # joint6: wrist angle (well within [-0.018, 3.753])
                0.785,         # joint7: π/4
                0.04,          # finger1: open
                0.04,          # finger2: open
            ])
            self.table_scene.initialize(env_idx, qpos_0=better_qpos)

            # Raise table to be reachable
            table_pose = self.table_scene.table.pose
            table_p = np.asarray(table_pose.p).ravel()
            table_q = table_pose.q
            if torch.is_tensor(table_q):
                table_q = table_q.cpu().numpy().ravel()
            else:
                table_q = np.asarray(table_q).ravel()

            # Apply height offset to bring table into robot's workspace
            new_table_p = np.array([table_p[0], table_p[1], table_p[2] + self.table_height_offset], dtype=np.float32)
            new_table_q = np.array(table_q, dtype=np.float32)
            self.table_scene.table.set_pose(sapien.Pose(p=new_table_p, q=new_table_q))

            # Place plate above table and let it drop to settle properly
            # Compute table top Z so we place objects reliably on the surface
            # Robustly read the table top Z coordinate. table.pose.p may be a
            # numpy array, a torch tensor, or a batched 1x3 tensor; convert to
            # a flat numpy array and take the Z (last) component.
            table_p_arr = np.asarray(self.table_scene.table.pose.p).ravel()
            table_z = float(table_p_arr[-1])
            table_top_z = table_z + float(self.table_scene.table_height)

            # Place the plate flat on the table with the rim facing up
            xyz = torch.zeros((b, 3), device=device)
            xyz[:, 0] = -0.35  # Closer to robot reach zone
            xyz[:, 1] = -0.15  # More centered for easier access
            plate_half_height = self._plate_total_height / 2.0
            xyz[:, 2] = table_top_z

            # Keep plate horizontal on the table with identity quaternion
            # The circular face should be parallel to the table surface with normal pointing up
            flat_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(b, 1)

            plate_pose = Pose.create_from_pq(p=xyz, q=flat_quat)
            self.plate.set_pose(plate_pose)

            rack_pos = torch.zeros((b, 3), device=device)
            rack_pos[:] = torch.tensor(self._rack_position, device=device)
            # Set rack z so its bottom rests on the table surface (use half height)
            rack_pos[:, 2] = table_top_z + float(self._rack_extent[2])
            rack_pose = Pose.create_from_pq(p=rack_pos)
            self.dish_rack.set_pose(rack_pose)

            # Position plate support pedestal under plate if it exists
            if self.plate_support is not None:
                support_pos = xyz.clone()
                support_pos[:, 2] = self._plate_support_height / 2  # Center of cylinder
                support_pose = Pose.create_from_pq(p=support_pos)
                self.plate_support.set_pose(support_pose)

            # Let the plate settle on the table for stable physics
            for _ in range(50):
                self.scene.step()

            # Force the plate back to its intended flat pose in case it rolled during settling.
            xyz[:, 2] = table_top_z + plate_half_height + self._plate_spawn_buffer
            plate_pose = Pose.create_from_pq(p=xyz, q=flat_quat)
            self.plate.set_pose(plate_pose)

            # Zero velocities after settling (but keep the settled pose to avoid interpenetration)
            zero_velocity = torch.zeros((b, 3), device=device)
            self.plate.set_linear_velocity(zero_velocity)
            self.plate.set_angular_velocity(zero_velocity)

    def evaluate(self):
        plate_pos = self.plate.pose.p
        rack_pos = self.dish_rack.pose.p
        target_offset = torch.tensor(
            self._plate_goal_offset, device=self.device, dtype=plate_pos.dtype
        )
        goal_pos = rack_pos + target_offset
        plate_to_goal = torch.linalg.norm(plate_pos - goal_pos, dim=1)

        rot_mats = quaternion_to_matrix(self.plate.pose.q)
        plate_norm = rot_mats[..., 2]
        plate_vertical = torch.abs(plate_norm[..., 2]) <= 0.35

        is_grasped = self.agent.is_grasping(self.plate)
        is_static = self.plate.is_static(lin_thresh=0.02, ang_thresh=0.4)

        # Check that plate is above table surface (not clipping through)
        # Table surface is at z=0, plate bottom should be at least at z > -0.01
        above_table = plate_pos[:, 2] > -0.01

        close_to_rack = plate_to_goal <= 0.08
        success = close_to_rack & plate_vertical & (~is_grasped) & is_static & above_table

        return {
            "success": success,
            "plate_close_to_goal": close_to_rack,
            "plate_vertical": plate_vertical,
            "is_static": is_static,
            "is_grasped": is_grasped,
            "above_table": above_table,
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
            obs.update(
                plate_to_goal=plate_pose.p - rack_pose.p,
                tcp_to_plate=plate_pose.p - self.agent.tcp_pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict) -> torch.Tensor:
        """Compute dense reward for the task."""
        plate_pos = self.plate.pose.p
        rack_pos = self.dish_rack.pose.p
        target_offset = torch.tensor(
            self._plate_goal_offset, device=self.device, dtype=plate_pos.dtype
        )
        goal_pos = rack_pos + target_offset
        plate_to_goal_dist = torch.linalg.norm(plate_pos - goal_pos, dim=1)

        # Distance reward
        reaching_reward = 1.0 - torch.tanh(5.0 * plate_to_goal_dist)

        # Orientation reward (plate should be vertical)
        rot_mats = quaternion_to_matrix(self.plate.pose.q)
        plate_norm = rot_mats[..., 2]
        vertical_alignment = torch.abs(plate_norm[..., 2])
        orientation_reward = 1.0 - vertical_alignment

        # Gripper release reward
        is_grasped = self.agent.is_grasping(self.plate)
        close_to_goal = plate_to_goal_dist <= 0.08
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
