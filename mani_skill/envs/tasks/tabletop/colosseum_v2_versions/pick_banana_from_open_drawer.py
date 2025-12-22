from typing import Any, Dict, Union
import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Panda, Fetch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.envs.distraction_set import DistractionSet

CABINET_COLLISION_BIT = 29


@register_env(
    "PickBananaFromOpenDrawer-v1",
    asset_download_ids=["partnet_mobility_cabinet"],
    max_episode_steps=50,
)
class PickBananaFromOpenDrawerEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]
    agent: Union[Panda, Fetch]
    
    CABINET_MODEL_ID = 1027
    CABINET_X_LIMS = [0.15, 0.25]
    CABINET_Y_LIMS = [-0.05, 0.05]
    LIFT_HEIGHT = 0.30
    
    def __init__(
        self, 
        *args, 
        robot_uids="panda_wristcam", 
        robot_init_qpos_noise=0.02,
        **kwargs
    ):
        distraction_set: Union[DistractionSet, dict] = kwargs.pop("distraction_set")
        self._distraction_set: DistractionSet = DistractionSet(**distraction_set) if isinstance(distraction_set, dict) else distraction_set
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.0, 0, 0.8], target=[0.15, 0, 0.2])
        pose2 = sapien_utils.look_at(eye=[-0.3, 0.3, 0.6], target=[0.15, 0, 0.2])
        return [
            CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100),
            CameraConfig("side_camera", pose2, 128, 128, np.pi / 2, 0.01, 100)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-0.6, -0.7, 0.6], [0.15, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        
        sapien.set_log_level("off")
        self._load_cabinet()
        sapien.set_log_level("warn")
        
        self._load_banana()

    def _load_cabinet(self):
        cabinet_builder = articulations.get_articulation_builder(
            self.scene, f"partnet-mobility:{self.CABINET_MODEL_ID}"
        )
        cabinet_builder.initial_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
        self.cabinet = cabinet_builder.build(name=f"cabinet-{self.CABINET_MODEL_ID}")
        
        for link in self.cabinet.links:
            link.set_collision_group_bit(
                group=2, bit_idx=CABINET_COLLISION_BIT, bit=1
            )
        
        self.drawer_joint = None
        self.drawer_link = None
        for joint, link in zip(self.cabinet.joints, self.cabinet.links):
            if joint.type[0] == "prismatic":
                self.drawer_joint = joint
                self.drawer_link = link
                break
        
        self.drawer_qlimits = self.drawer_joint.limits
        self.drawer_qmin = self.drawer_qlimits[0, 0] if len(self.drawer_qlimits.shape) > 1 else self.drawer_qlimits[0]
        self.drawer_qmax = self.drawer_qlimits[0, 1] if len(self.drawer_qlimits.shape) > 1 else self.drawer_qlimits[1]
        
        collision_mesh = self.cabinet.get_first_collision_mesh()
        self.cabinet_z = -collision_mesh.bounding_box.bounds[0, 2]

    def _load_banana(self):
        # TODO(@orhun): Throw an error if the banana is not found.
        # TODO(@orhun): Add a command line
        try:
            banana_builder = actors.get_actor_builder(self.scene, id="ycb:011_banana")
            banana_builder.initial_pose = sapien.Pose(p=[0.15, 0, 0.1])
            self.banana = banana_builder.build(name="banana")
        except Exception as e:
            builder = self.scene.create_actor_builder()
            builder.add_capsule_visual(
                radius=0.015,
                half_length=0.05,
                color=[1.0, 0.9, 0.1, 1],
                pose=sapien.Pose(q=[0.707, 0, 0.707, 0])
            )
            builder.add_capsule_collision(
                radius=0.015,
                half_length=0.05,
                pose=sapien.Pose(q=[0.707, 0, 0.707, 0])
            )
            builder.initial_pose = sapien.Pose(p=[0.15, 0, 0.1])
            self.banana = builder.build(name="banana")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            
            qpos_0 = np.array([-0.13595445, -1.2611351, 0.24094589, -2.9000182, 
                              2.5728698, 3.0259767, 0.029944034, 0.039999813, 0.03999985])
            self.table_scene.initialize(env_idx, table_z_rotation_angle=np.pi, qpos_0=qpos_0)
            
            cabinet_xy = torch.zeros((b, 3))
            cabinet_x_range = self.CABINET_X_LIMS[1] - self.CABINET_X_LIMS[0]
            cabinet_y_range = self.CABINET_Y_LIMS[1] - self.CABINET_Y_LIMS[0]
            cabinet_xy[:, 0] = torch.rand(b) * cabinet_x_range + self.CABINET_X_LIMS[0]
            cabinet_xy[:, 1] = torch.rand(b) * cabinet_y_range + self.CABINET_Y_LIMS[0]
            cabinet_xy[:, 2] = self.cabinet_z
            
            self.cabinet.set_pose(Pose.create_from_pq(p=cabinet_xy))
            
            open_amount = 0.7
            drawer_qpos = self.drawer_qmin + (self.drawer_qmax - self.drawer_qmin) * open_amount
            current_qpos = self.cabinet.get_qpos()
            for i, joint in enumerate(self.cabinet.joints):
                if joint == self.drawer_joint:
                    current_qpos[env_idx, i] = drawer_qpos
                    break
            self.cabinet.set_qpos(current_qpos[env_idx])
            self.cabinet.set_qvel(self.cabinet.qpos[env_idx] * 0)
            
            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene.px.step()
                self.scene._gpu_fetch_all()
            
            drawer_interior_depth = 0.15 * open_amount
            drawer_bottom_height = self.cabinet_z + 0.05
            
            banana_xyz = torch.zeros((b, 3))
            banana_xyz[:, 0] = cabinet_xy[:, 0] + torch.rand(b) * 0.1 - 0.05
            banana_xyz[:, 1] = cabinet_xy[:, 1] + drawer_interior_depth / 2
            banana_xyz[:, 2] = drawer_bottom_height + 0.02
            
            qs = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
            )
            
            self.banana.set_pose(Pose.create_from_pq(p=banana_xyz, q=qs))
            
            self.drawer_open_qpos = drawer_qpos
            self.drawer_top_height = drawer_bottom_height + 0.1

    def evaluate(self):
        is_banana_grasped = self.agent.is_grasping(self.banana)
        
        banana_height = self.banana.pose.p[..., 2]
        is_lifted = banana_height > self.LIFT_HEIGHT
        
        is_banana_static = self.banana.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        
        success = is_banana_grasped & is_lifted & is_banana_static
        
        return {
            "is_banana_grasped": is_banana_grasped,
            "is_lifted": is_lifted, 
            "is_banana_static": is_banana_static,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                banana_pose=self.banana.pose.raw_pose,
                tcp_to_banana_pos=self.banana.pose.p - self.agent.tcp.pose.p,
                cabinet_pose=self.cabinet.pose.raw_pose,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        reward = torch.zeros(self.num_envs, device=self.device)

        tcp_pos = self.agent.tcp.pose.p
        banana_pos = self.banana.pose.p

        tcp_to_banana_dist = torch.linalg.norm(tcp_pos - banana_pos, axis=1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_banana_dist)
        
        is_grasping = self.agent.is_grasping(self.banana)
        reaching_reward[is_grasping] = 1.0
        reward += reaching_reward

        grasp_reward = is_grasping.float() * 0.5
        reward += grasp_reward

        banana_height = banana_pos[:, 2]
        lift_progress = torch.clamp((banana_height - self.drawer_top_height) / 0.15, 0.0, 1.0)
        lift_reward = lift_progress * is_grasping.float() * 2.0
        reward += lift_reward

        reward[info["success"]] = 5.0

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 5.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
        