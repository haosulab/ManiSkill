"""
Base environment for Bridge dataset environments
"""
import os
from typing import Dict, List, Literal

import numpy as np
import sapien
import torch
from sapien.physx import PhysxMaterial
from transforms3d.quaternions import quat2mat

from mani_skill import ASSET_DIR
from mani_skill.agents.controllers.pd_ee_pose import PDEEPoseControllerConfig
from mani_skill.agents.controllers.pd_joint_pos import PDJointPosMimicControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.widowx.widowx import WidowX250S
from mani_skill.envs.tasks.digital_twins.base_env import BaseDigitalTwinEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, io_utils, sapien_utils
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig

BRIDGE_DATASET_ASSET_PATH = ASSET_DIR / "tasks/bridge_v2_real2sim_dataset/"
# Real2Sim tuned WidowX250S robot
@register_agent(asset_download_ids=["widowx250s"])
class WidowX250SBridgeDatasetFlatTable(WidowX250S):
    uid = "widowx250s_bridgedataset_flat_table"
    arm_joint_names = [
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
    ]
    gripper_joint_names = ["left_finger", "right_finger"]

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="3rd_view_camera",  # the camera used in the Bridge dataset
                pose=sapien.Pose(
                    [0.00, -0.16, 0.36],
                    [0.8992917, -0.09263245, 0.35892478, 0.23209205],
                ),
                width=640,
                height=480,
                entity_uid="base_link",
                intrinsic=np.array(
                    [[623.588, 0, 319.501], [0, 623.588, 239.545], [0, 0, 1]]
                ),  # logitech C920
            ),
        ]

    arm_stiffness = [
        1169.7891719504198,
        730.0,
        808.4601346394447,
        1229.1299089624076,
        1272.2760456418862,
        1056.3326605132252,
    ]
    arm_damping = [
        330.0,
        180.0,
        152.12036565582588,
        309.6215302722146,
        201.04998711007383,
        269.51458932695414,
    ]

    arm_force_limit = [200, 200, 100, 100, 100, 100]
    arm_friction = 0.0
    arm_vel_limit = 1.5
    arm_acc_limit = 2.0

    gripper_stiffness = 1000
    gripper_damping = 200
    gripper_pid_stiffness = 1000
    gripper_pid_damping = 200
    gripper_pid_integral = 300
    gripper_force_limit = 60
    gripper_vel_limit = 0.12
    gripper_acc_limit = 0.50
    gripper_jerk_limit = 5.0

    @property
    def _controller_configs(self):
        arm_common_kwargs = dict(
            joint_names=self.arm_joint_names,
            pos_lower=-1.0,  # dummy limit, which is unused since normalize_action=False
            pos_upper=1.0,
            rot_lower=-np.pi / 2,
            rot_upper=np.pi / 2,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            friction=self.arm_friction,
            ee_link="ee_gripper_link",
            urdf_path=self.urdf_path,
            normalize_action=False,
        )
        arm_pd_ee_target_delta_pose_align2 = PDEEPoseControllerConfig(
            **arm_common_kwargs, use_target=True
        )

        extra_gripper_clearance = 0.001  # since real gripper is PID, we use extra clearance to mitigate PD small errors; also a trick to have force when grasping
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            joint_names=self.gripper_joint_names,
            lower=0.015 - extra_gripper_clearance,
            upper=0.037 + extra_gripper_clearance,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            normalize_action=True,
            drive_mode="force",
        )
        controller = dict(
            arm=arm_pd_ee_target_delta_pose_align2, gripper=gripper_pd_joint_pos
        )
        return dict(arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos=controller)


# Tuned for the sink setup
@register_agent(asset_download_ids=["widowx250s"])
class WidowX250SBridgeDatasetSink(WidowX250SBridgeDatasetFlatTable):
    uid = "widowx250s_bridgedataset_sink"

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="3rd_view_camera",  # the camera used for real evaluation for the sink setup
                pose=sapien.Pose(
                    [-0.00300001, -0.21, 0.39], [-0.907313, 0.0782, -0.36434, -0.194741]
                ),
                entity_uid="base_link",
                width=640,
                # fov=1.5,
                height=480,
                near=0.01,
                far=10,
                intrinsic=np.array(
                    [[623.588, 0, 319.501], [0, 623.588, 239.545], [0, 0, 1]]
                ),
            )
        ]


class BaseBridgeEnv(BaseDigitalTwinEnv):
    """Base Digital Twin environment for digital twins of the BridgeData v2"""

    MODEL_JSON = "info_bridge_custom_v0.json"
    SUPPORTED_OBS_MODES = ["rgb+segmentation"]
    SUPPORTED_REWARD_MODES = ["none"]
    scene_setting: Literal["flat_table", "sink"] = "flat_table"

    obj_static_friction = 0.5
    obj_dynamic_friction = 0.5

    def __init__(
        self,
        obj_names: List[str],
        xyz_configs: torch.Tensor,
        quat_configs: torch.Tensor,
        **kwargs,
    ):
        self.objs: Dict[str, Actor] = dict()
        self.obj_names = obj_names
        self.source_obj_name = obj_names[0]
        self.target_obj_name = obj_names[1]
        self.xyz_configs = xyz_configs
        self.quat_configs = quat_configs
        if self.scene_setting == "flat_table":
            self.rgb_overlay_paths = {
                "3rd_view_camera": str(
                    BRIDGE_DATASET_ASSET_PATH / "real_inpainting/bridge_real_eval_1.png"
                )
            }
            robot_cls = WidowX250SBridgeDatasetFlatTable
        elif self.scene_setting == "sink":
            self.rgb_overlay_paths = {
                "3rd_view_camera": str(
                    BRIDGE_DATASET_ASSET_PATH / "real_inpainting/bridge_sink.png"
                )
            }
            robot_cls = WidowX250SBridgeDatasetSink

        self.model_db: Dict[str, Dict] = io_utils.load_json(
            BRIDGE_DATASET_ASSET_PATH / "custom/" / self.MODEL_JSON
        )
        super().__init__(
            robot_uids=robot_cls,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=500, control_freq=5, spacing=20)

    @property
    def _default_human_render_camera_configs(self):
        sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera",
            pose=sapien.Pose(
                [0.00, -0.16, 0.336], [0.909182, -0.0819809, 0.347277, 0.214629]
            ),
            width=512,
            height=512,
            intrinsic=np.array(
                [[623.588, 0, 319.501], [0, 623.588, 239.545], [0, 0, 1]]
            ),
            near=0.01,
            far=100,
            mount=self.agent.robot.links_map["base_link"],
        )

    def _build_actor_helper(
        self,
        model_id: str,
        scale: float = 1,
        kinematic: bool = False,
        initial_pose: Pose = None,
    ):
        """helper function to build actors by ID directly and auto configure physical materials"""
        density = self.model_db[model_id].get("density", 1000)
        physical_material = PhysxMaterial(
            static_friction=self.obj_static_friction,
            dynamic_friction=self.obj_dynamic_friction,
            restitution=0.0,
        )
        builder = self.scene.create_actor_builder()
        model_dir = BRIDGE_DATASET_ASSET_PATH / "custom" / "models" / model_id

        collision_file = str(model_dir / "collision.obj")
        builder.add_multiple_convex_collisions_from_file(
            filename=collision_file,
            scale=[scale] * 3,
            material=physical_material,
            density=density,
        )

        visual_file = str(model_dir / "textured.obj")
        if not os.path.exists(visual_file):
            visual_file = str(model_dir / "textured.dae")
            if not os.path.exists(visual_file):
                visual_file = str(model_dir / "textured.glb")
        builder.add_visual_from_file(filename=visual_file, scale=[scale] * 3)
        if initial_pose is not None:
            builder.initial_pose = initial_pose
        if kinematic:
            actor = builder.build_kinematic(name=model_id)
        else:
            actor = builder.build(name=model_id)
        return actor

    def _load_lighting(self, options: dict):
        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light(
            [0, 0, -1],
            [2.2, 2.2, 2.2],
            shadow=False,
            shadow_scale=5,
            shadow_map_size=2048,
        )
        self.scene.add_directional_light([-1, -0.5, -1], [0.7, 0.7, 0.7])
        self.scene.add_directional_light([1, 1, -1], [0.7, 0.7, 0.7])

    def _load_agent(self, options: dict):
        super()._load_agent(
            options, sapien.Pose(p=[0.127, 0.060, 0.85], q=[0, 0, 0, 1])
        )

    def _load_scene(self, options: dict):
        # original SIMPLER envs always do this? except for open drawer task
        for i in range(self.num_envs):
            sapien_utils.set_articulation_render_material(
                self.agent.robot._objs[i], specular=0.9, roughness=0.3
            )

        # load background
        builder = self.scene.create_actor_builder()
        scene_pose = sapien.Pose(q=[0.707, 0.707, 0, 0])
        scene_offset = np.array([-2.0634, -2.8313, 0.0])
        if self.scene_setting == "flat_table":
            scene_file = str(BRIDGE_DATASET_ASSET_PATH / "stages/bridge_table_1_v1.glb")

        elif self.scene_setting == "sink":
            scene_file = str(BRIDGE_DATASET_ASSET_PATH / "stages/bridge_table_1_v2.glb")
        builder.add_nonconvex_collision_from_file(scene_file, pose=scene_pose)
        builder.add_visual_from_file(scene_file, pose=scene_pose)

        builder.initial_pose = sapien.Pose(-scene_offset)
        builder.build_static(name="arena")

        for name in self.obj_names:
            self.objs[name] = self._build_actor_helper(name)

        self.xyz_configs = common.to_tensor(self.xyz_configs, device=self.device).to(
            torch.float32
        )
        self.quat_configs = common.to_tensor(self.quat_configs, device=self.device).to(
            torch.float32
        )

        if self.scene_setting == "sink":
            self.sink = self._build_actor_helper(
                "sink",
                kinematic=True,
                initial_pose=sapien.Pose([-0.16, 0.13, 0.88], [1, 0, 0, 0]),
            )
        # model scales
        model_scales = None
        if model_scales is None:
            model_scales = dict()
            for model_id in [self.source_obj_name, self.target_obj_name]:
                this_available_model_scales = self.model_db[model_id].get(
                    "scales", None
                )
                if this_available_model_scales is None:
                    model_scales.append(1.0)
                else:
                    # TODO (stao): use the batched RNG
                    model_scales[model_id] = self.np_random.choice(
                        this_available_model_scales
                    )
        self.episode_model_scales = model_scales
        model_bbox_sizes = dict()
        for model_id in [self.source_obj_name, self.target_obj_name]:
            model_info = self.model_db[model_id]
            model_scale = self.episode_model_scales[model_id]
            if "bbox" in model_info:
                bbox = model_info["bbox"]
                bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])
                model_bbox_sizes[model_id] = common.to_tensor(
                    bbox_size * model_scale, device=self.device
                )
            else:
                raise ValueError(f"Model {model_id} does not have bbox info.")
        self.episode_model_bbox_sizes = model_bbox_sizes

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # NOTE: this part of code is not GPU parallelized
        with torch.device(self.device):
            b = len(env_idx)
            if "episode_id" in options:
                if isinstance(options["episode_id"], int):
                    options["episode_id"] = torch.tensor([options["episode_id"]])
                    assert len(options["episode_id"]) == b
                pos_episode_ids = (
                    options["episode_id"]
                    % (len(self.xyz_configs) * len(self.quat_configs))
                ) // len(self.quat_configs)
                quat_episode_ids = options["episode_id"] % len(self.quat_configs)
            else:
                pos_episode_ids = torch.randint(0, len(self.xyz_configs), size=(b,))
                quat_episode_ids = torch.randint(0, len(self.quat_configs), size=(b,))
            for i, actor in enumerate(self.objs.values()):
                xyz = self.xyz_configs[pos_episode_ids, i]
                actor.set_pose(
                    Pose.create_from_pq(p=xyz, q=self.quat_configs[quat_episode_ids, i])
                )
            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()
            self._settle(0.5)
            if self.gpu_sim_enabled:
                self.scene._gpu_fetch_all()
            # Some objects need longer time to settle
            lin_vel, ang_vel = 0.0, 0.0
            for obj_name, obj in self.objs.items():
                lin_vel += torch.linalg.norm(obj.linear_velocity)
                ang_vel += torch.linalg.norm(obj.angular_velocity)
            if lin_vel > 1e-3 or ang_vel > 1e-2:
                if self.gpu_sim_enabled:
                    self.scene._gpu_apply_all()
                self._settle(6)
                if self.gpu_sim_enabled:
                    self.scene._gpu_fetch_all()
            # measured values for bridge dataset
            if self.scene_setting == "flat_table":
                qpos = np.array(
                    [
                        -0.01840777,
                        0.0398835,
                        0.22242722,
                        -0.00460194,
                        1.36524296,
                        0.00153398,
                        0.037,
                        0.037,
                    ]
                )

                self.agent.robot.set_pose(
                    sapien.Pose([0.147, 0.028, 0.870], q=[0, 0, 0, 1])
                )
            elif self.scene_setting == "sink":
                qpos = np.array(
                    [
                        -0.2600599,
                        -0.12875618,
                        0.04461369,
                        -0.00652761,
                        1.7033415,
                        -0.26983038,
                        0.037,
                        0.037,
                    ]
                )
                self.agent.robot.set_pose(
                    sapien.Pose([0.127, 0.060, 0.85], q=[0, 0, 0, 1])
                )
            self.agent.reset(init_qpos=qpos)

            # figure out object bounding boxes after settling. This is used to determine if an object is near the target object
            self.episode_source_obj_xyz_after_settle = self.objs[
                self.source_obj_name
            ].pose.p
            self.episode_target_obj_xyz_after_settle = self.objs[
                self.target_obj_name
            ].pose.p
            self.episode_obj_xyzs_after_settle = {
                obj_name: self.objs[obj_name].pose.p for obj_name in self.objs.keys()
            }
            self.episode_source_obj_bbox_world = self.episode_model_bbox_sizes[
                self.source_obj_name
            ].float()
            self.episode_target_obj_bbox_world = self.episode_model_bbox_sizes[
                self.target_obj_name
            ].float()
            self.episode_source_obj_bbox_world = (
                rotation_conversions.quaternion_to_matrix(
                    self.objs[self.source_obj_name].pose.q
                )
                @ self.episode_source_obj_bbox_world[..., None]
            )[0, :, 0]
            """source object bbox size (3, )"""
            self.episode_target_obj_bbox_world = (
                rotation_conversions.quaternion_to_matrix(
                    self.objs[self.target_obj_name].pose.q
                )
                @ self.episode_target_obj_bbox_world[..., None]
            )[0, :, 0]
            """target object bbox size (3, )"""

            # stats to track
            self.consecutive_grasp = torch.zeros((b,), dtype=torch.int32)
            self.episode_stats = dict(
                # all_obj_keep_height=torch.zeros((b,), dtype=torch.bool),
                moved_correct_obj=torch.zeros((b,), dtype=torch.bool),
                moved_wrong_obj=torch.zeros((b,), dtype=torch.bool),
                # near_tgt_obj=torch.zeros((b,), dtype=torch.bool),
                is_src_obj_grasped=torch.zeros((b,), dtype=torch.bool),
                # is_closest_to_tgt=torch.zeros((b,), dtype=torch.bool),
                consecutive_grasp=torch.zeros((b,), dtype=torch.bool),
            )

    def _settle(self, t: int = 0.5):
        """run the simulation for some steps to help settle the objects"""
        sim_steps = int(self.sim_freq * t / self.control_freq)
        for _ in range(sim_steps):
            self.scene.step()

    def _evaluate(
        self,
        success_require_src_completely_on_target=True,
        z_flag_required_offset=0.02,
        **kwargs,
    ):
        source_object = self.objs[self.source_obj_name]
        target_object = self.objs[self.target_obj_name]
        source_obj_pose = source_object.pose
        target_obj_pose = target_object.pose

        # whether moved the correct object
        source_obj_xy_move_dist = torch.linalg.norm(
            self.episode_source_obj_xyz_after_settle[:, :2] - source_obj_pose.p[:, :2],
            dim=1,
        )
        other_obj_xy_move_dist = []
        for obj_name in self.objs.keys():
            obj = self.objs[obj_name]
            obj_xyz_after_settle = self.episode_obj_xyzs_after_settle[obj_name]
            if obj.name == self.source_obj_name:
                continue
            other_obj_xy_move_dist.append(
                torch.linalg.norm(
                    obj_xyz_after_settle[:, :2] - obj.pose.p[:, :2], dim=1
                )
            )
        # moved_correct_obj = (source_obj_xy_move_dist > 0.03) and (
        #     all([x < source_obj_xy_move_dist for x in other_obj_xy_move_dist])
        # )
        # moved_wrong_obj = any([x > 0.03 for x in other_obj_xy_move_dist]) and any(
        #     [x > source_obj_xy_move_dist for x in other_obj_xy_move_dist]
        # )
        # moved_correct_obj = False
        # moved_wrong_obj = False

        # whether the source object is grasped
        is_src_obj_grasped = self.agent.is_grasping(source_object)
        # if is_src_obj_grasped:
        self.consecutive_grasp += is_src_obj_grasped
        self.consecutive_grasp[is_src_obj_grasped == 0] = 0
        consecutive_grasp = self.consecutive_grasp >= 5

        # whether the source object is on the target object based on bounding box position
        tgt_obj_half_length_bbox = (
            self.episode_target_obj_bbox_world / 2
        )  # get half-length of bbox xy diagonol distance in the world frame at timestep=0
        src_obj_half_length_bbox = self.episode_source_obj_bbox_world / 2

        pos_src = source_obj_pose.p
        pos_tgt = target_obj_pose.p
        offset = pos_src - pos_tgt
        xy_flag = (
            torch.linalg.norm(offset[:, :2], dim=1)
            <= torch.linalg.norm(tgt_obj_half_length_bbox[:2]) + 0.003
        )
        z_flag = (offset[:, 2] > 0) & (
            offset[:, 2] - tgt_obj_half_length_bbox[2] - src_obj_half_length_bbox[2]
            <= z_flag_required_offset
        )
        src_on_target = xy_flag & z_flag
        # src_on_target = False

        if success_require_src_completely_on_target:
            # whether the source object is on the target object based on contact information
            contact_forces = self.scene.get_pairwise_contact_forces(
                source_object, target_object
            )
            net_forces = torch.linalg.norm(contact_forces, dim=1)
            src_on_target = src_on_target & (net_forces > 0.05)

        success = src_on_target

        # self.episode_stats["moved_correct_obj"] = moved_correct_obj
        # self.episode_stats["moved_wrong_obj"] = moved_wrong_obj
        self.episode_stats["src_on_target"] = src_on_target
        self.episode_stats["is_src_obj_grasped"] = (
            self.episode_stats["is_src_obj_grasped"] | is_src_obj_grasped
        )
        self.episode_stats["consecutive_grasp"] = (
            self.episode_stats["consecutive_grasp"] | consecutive_grasp
        )

        return dict(**self.episode_stats, success=success)

    def is_final_subtask(self):
        # whether the current subtask is the final one, only meaningful for long-horizon tasks
        return True
