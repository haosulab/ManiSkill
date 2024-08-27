"""
Base environment for Bridge dataset environments
"""
from typing import Dict, List, Literal

import numpy as np
import sapien
import torch
from sapien.physx import PhysxMaterial

from mani_skill import ASSET_DIR
from mani_skill.agents.controllers.pd_ee_pose import PDEEPoseControllerConfig
from mani_skill.agents.controllers.pd_joint_pos import PDJointPosMimicControllerConfig
from mani_skill.agents.robots.widowx.widowx import WidowX250S
from mani_skill.envs.tasks.digital_twins.base_env import BaseDigitalTwinEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, io_utils, sapien_utils
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig


# Real2Sim tuned WidowX250S robot
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
class WidowX250SBridgeDatasetSink(WidowX250S):
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
                height=480,
                intrinsic=np.array(
                    [[623.588, 0, 319.501], [0, 623.588, 239.545], [0, 0, 1]]
                ),
            )
        ]


class BaseBridgeEnv(BaseDigitalTwinEnv):
    """Base Digital Twin environment for digital twins of the BridgeData v2"""

    SUPPORTED_OBS_MODES = ["rgb+segmentation"]
    SUPPORTED_REWARD_MODES = ["none"]
    scene_setting: Literal["flat_table", "sink"] = "flat_table"
    rgb_overlay_cameras = ["3rd_view_camera"]
    rgb_overlay_path = ""
    scene_table_height: float = 0.87
    objs: Dict[str, Actor] = dict()

    obj_static_friction = 0.5
    obj_dynamic_friction = 0.5

    def __init__(
        self,
        obj_names: List[str],
        xyz_configs: torch.Tensor,
        quat_configs: torch.Tensor,
        **kwargs
    ):
        self.obj_names = obj_names
        self.xyz_configs = xyz_configs
        self.quat_configs = quat_configs
        if self.scene_setting == "flat_table":
            self.rgb_overlay_path = str(
                ASSET_DIR
                / "tasks/bridge_dataset/real_inpainting/bridge_real_eval_1.png"
            )
            robot_cls = WidowX250SBridgeDatasetFlatTable
        elif self.scene_setting == "sink":
            self.rgb_overlay_path = str(
                ASSET_DIR / "tasks/bridge_dataset/real_inpainting/bridge_sink.png"
            )
            robot_cls = WidowX250SBridgeDatasetSink

        self.model_db: Dict[str, Dict] = io_utils.load_json(
            ASSET_DIR / "tasks/bridge_dataset/custom/info_bridge_custom_v0.json"
        )
        super().__init__(
            robot_uids=robot_cls,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=500, control_freq=5)

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
    ):
        density = self.model_db[model_id].get("density", 1000)
        return super()._build_actor_helper(
            model_id,
            scale,
            physical_material=PhysxMaterial(
                static_friction=self.obj_static_friction,
                dynamic_friction=self.obj_dynamic_friction,
                restitution=0.0,
            ),
            density=density,
            root_dir=ASSET_DIR / "tasks/bridge_dataset/custom",
        )

    def _load_scene(self, options: dict):
        # load background
        builder = self.scene.create_actor_builder()
        scene_pose = sapien.Pose(q=[0.707, 0.707, 0, 0])
        scene_offset = np.array([-2.0634, -2.8313, 0.0])
        if self.scene_setting == "flat_table":
            scene_file = str(
                ASSET_DIR / "tasks/bridge_dataset/stages/bridge_table_1_v1.glb"
            )

        elif self.scene_setting == "sink":
            scene_file = str(
                ASSET_DIR / "tasks/bridge_dataset/stages/bridge_table_1_v2.glb"
            )
        builder.add_nonconvex_collision_from_file(scene_file, pose=scene_pose)
        builder.add_visual_from_file(scene_file, pose=scene_pose)

        builder.initial_pose = sapien.Pose(-scene_offset)
        builder.build_static(name="arena")

        for name in self.obj_names:
            self.objs[name] = self._build_actor_helper(name)

        self.xyz_configs = common.to_tensor(self.xyz_configs).to(torch.float32)
        self.quat_configs = common.to_tensor(self.quat_configs).to(torch.float32)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            pos_episode_ids = torch.randint(0, len(self.xyz_configs), size=(b,))
            quat_episode_ids = torch.randint(0, len(self.quat_configs), size=(b,))
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
                    sapien.Pose([0.147, 0.070, 0.85], q=[0, 0, 0, 1])
                )
            self.agent.robot.set_qpos(qpos)

            for i, actor in enumerate(self.objs.values()):
                xyz = self.xyz_configs[pos_episode_ids, i]
                actor.set_pose(
                    Pose.create_from_pq(p=xyz, q=self.quat_configs[quat_episode_ids, i])
                )

    def is_final_subtask(self):
        # whether the current subtask is the final one, only meaningful for long-horizon tasks
        return True
