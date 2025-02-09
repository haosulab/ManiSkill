from typing import Any, Dict, List, Union

import numpy as np
import sapien
import torch
import torch.nn.functional as F

from mani_skill import ASSET_DIR
from mani_skill.agents.robots import AllegroHandRightTouch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.building.actors import build_cube
from mani_skill.utils.geometry.rotation_conversions import quaternion_apply
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose, vectorize_pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


class RotateSingleObjectInHand(BaseEnv):
    agent: Union[AllegroHandRightTouch]
    _clearance = 0.003
    hand_init_height = 0.25

    def __init__(
        self,
        *args,
        robot_init_qpos_noise=0.02,
        obj_init_pos_noise=0.02,
        difficulty_level: int = -1,
        num_envs=1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.obj_init_pos_noise = obj_init_pos_noise
        self.obj_heights: torch.Tensor = torch.Tensor()
        if (
            not isinstance(difficulty_level, int)
            or difficulty_level >= 4
            or difficulty_level < 0
        ):
            raise ValueError(
                f"Difficulty level must be a int within 0-3, but get {difficulty_level}"
            )
        self.difficulty_level = difficulty_level
        if self.difficulty_level >= 2:
            if reconfiguration_freq is None:
                if num_envs == 1:
                    reconfiguration_freq = 1
                else:
                    reconfiguration_freq = 0
        super().__init__(
            *args,
            robot_uids="allegro_hand_right_touch",
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

        with torch.device(self.device):
            self.prev_unit_vector = torch.zeros((self.num_envs, 3))
            self.cum_rotation_angle = torch.zeros((self.num_envs,))

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=self.num_envs * max(1024, self.num_envs) * 8,
                max_rigid_patch_count=self.num_envs * max(1024, self.num_envs) * 2,
                found_lost_pairs_capacity=2**26,
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=[0.15, 0, 0.45], target=[-0.1, 0, self.hand_init_height]
        )
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.2, 0.4, 0.6], [0.0, 0.0, 0.3])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        obj_heights = []
        if self.difficulty_level == 0:
            self.obj = build_cube(
                self.scene,
                half_size=0.04,
                color=np.array([255, 255, 255, 255]) / 255,
                name="cube",
                body_type="dynamic",
            )
            obj_heights.append(0.03)
        elif self.difficulty_level == 1:
            half_sizes = (self._batched_episode_rng.randn() * 0.1 + 1) * 0.04
            self._objs: List[Actor] = []
            for i, half_size in enumerate(half_sizes):
                builder = self.scene.create_actor_builder()
                builder.add_box_collision(
                    half_size=[half_size] * 3,
                )
                builder.add_box_visual(
                    half_size=[half_size] * 3,
                    material=sapien.render.RenderMaterial(
                        base_color=np.array([255, 255, 255, 255]) / 255,
                    ),
                )
                builder.set_scene_idxs([i])
                self._objs.append(builder.build(name=f"cube-{i}"))
                obj_heights.append(half_size)
            self.obj = Actor.merge(self._objs, name="cube")
        elif self.difficulty_level >= 2:
            all_model_ids = np.array(
                list(
                    load_json(
                        ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json"
                    ).keys()
                )
            )
            model_ids = self._batched_episode_rng.choice(all_model_ids)
            self._objs: List[Actor] = []
            for i, model_id in enumerate(model_ids):
                builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
                builder.set_scene_idxs([i])
                self._objs.append(builder.build(name=f"{model_id}-{i}"))
            self.obj = Actor.merge(self._objs, name="ycb_object")
        else:
            raise ValueError(
                f"Difficulty level must be an int within 0-4, but get {self.difficulty_level}"
            )

        if self.difficulty_level < 2:
            # for levels 0 and 1 we already know object heights. For other levels we need to compute them
            self.obj_heights = common.to_tensor(obj_heights, device=self.device)

    def _after_reconfigure(self, options: dict):
        if self.difficulty_level >= 2:
            self.obj_heights = []
            for obj in self._objs:
                collision_mesh = obj.get_first_collision_mesh()
                # this value is used to set object pose so the bottom is at z=0
                self.obj_heights.append(-collision_mesh.bounding_box.bounds[0, 2])
            self.obj_heights = common.to_tensor(self.obj_heights, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._initialize_actors(env_idx)
        self._initialize_agent(env_idx)

    def _initialize_actors(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)
            # Initialize object pose
            self.table_scene.initialize(env_idx)
            pose = self.obj.pose
            new_pos = torch.randn((b, 3)) * self.obj_init_pos_noise
            # hand_init_height is robot hand position while the 0.03 is a margin to ensure
            new_pos[:, 2] = (
                torch.abs(new_pos[:, 2]) + self.hand_init_height + self.obj_heights
            )
            pose.raw_pose[:, 0:3] = new_pos
            pose.raw_pose[:, 3:7] = torch.tensor([[1, 0, 0, 0]])
            self.obj.set_pose(pose)

            # Initialize object axis
            if self.difficulty_level <= 2:
                axis = torch.ones((b,), dtype=torch.long) * 2
            else:
                axis = torch.randint(0, 3, (b,), dtype=torch.long)
            self.rot_dir = F.one_hot(axis, num_classes=3)

            # Sample a unit vector on the tangent plane of rotating axis
            vector_axis = (axis + 1) % 3
            vector = F.one_hot(vector_axis, num_classes=3)

            # Initialize task related cache
            self.unit_vector = vector
            self.prev_unit_vector = vector.clone()
            self.success_threshold = torch.pi * 4
            self.cum_rotation_angle = torch.zeros((b,))

            # Controller parameters
            stiffness = torch.tensor(self.agent.controller.config.stiffness)
            damping = torch.tensor(self.agent.controller.config.damping)
            force_limit = torch.tensor(self.agent.controller.config.force_limit)
            self.controller_param = (
                stiffness.expand(b, self.agent.robot.dof[0]),
                damping.expand(b, self.agent.robot.dof[0]),
                force_limit.expand(b, self.agent.robot.dof[0]),
            )

    def _initialize_agent(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            b = len(env_idx)
            dof = self.agent.robot.dof
            if isinstance(dof, torch.Tensor):
                dof = dof[0]
            init_qpos = torch.zeros((b, dof))
            self.agent.reset(init_qpos)
            self.agent.robot.set_pose(
                Pose.create_from_pq(
                    torch.tensor([0.0, 0, self.hand_init_height]),
                    torch.tensor([-0.707, 0, 0.707, 0]),
                )
            )

    def _get_obs_extra(self, info: Dict):
        with torch.device(self.device):
            obs = dict(rotate_dir=self.rot_dir)
            if self.obs_mode_struct.use_state:
                obs.update(
                    obj_pose=vectorize_pose(self.obj.pose),
                    obj_tip_vec=info["obj_tip_vec"].view(self.num_envs, 12),
                )
            return obs

    def evaluate(self, **kwargs) -> dict:
        with torch.device(self.device):
            # 1. rotation angle
            obj_pose = self.obj.pose
            new_unit_vector = quaternion_apply(obj_pose.q, self.unit_vector)
            new_unit_vector -= (
                torch.sum(new_unit_vector * self.rot_dir, dim=-1, keepdim=True)
                * self.rot_dir
            )
            new_unit_vector = new_unit_vector / torch.linalg.norm(
                new_unit_vector, dim=-1, keepdim=True
            )
            angle = torch.acos(
                torch.clip(
                    torch.sum(new_unit_vector * self.prev_unit_vector, dim=-1), 0, 1
                )
            )
            # We do not expect the rotation angle for a single step to be so large
            angle = torch.clip(angle, -torch.pi / 20, torch.pi / 20)
            self.prev_unit_vector = new_unit_vector

            # 2. object velocity
            obj_vel = torch.linalg.norm(self.obj.get_linear_velocity(), dim=-1)

            # 3. object falling
            obj_fall = (obj_pose.p[:, 2] < self.hand_init_height - 0.05).to(torch.bool)

            # 4. finger object distance
            tip_poses = [vectorize_pose(link.pose) for link in self.agent.tip_links]
            tip_poses = torch.stack(tip_poses, dim=1)  # (b, 4, 7)
            obj_tip_vec = tip_poses[..., :3] - obj_pose.p[:, None, :]  # (b, 4, 3)
            obj_tip_dist = torch.linalg.norm(obj_tip_vec, dim=-1)  # (b, 4)

            # 5. cum rotation angle
            self.cum_rotation_angle += angle
            success = self.cum_rotation_angle > self.success_threshold

            # 6. controller effort
            qpos_target = self.agent.controller._target_qpos
            qpos_error = qpos_target - self.agent.robot.qpos
            qvel = self.agent.robot.qvel
            qf = qpos_error * self.controller_param[0] - qvel * self.controller_param[1]
            qf = torch.clip(qf, -self.controller_param[2], self.controller_param[2])
            power = torch.sum(qf * qvel, dim=-1)

        return dict(
            rotation_angle=angle,
            obj_vel=obj_vel,
            obj_fall=obj_fall,
            obj_tip_vec=obj_tip_vec,
            obj_tip_dist=obj_tip_dist,
            success=success,
            qf=qf,
            power=power,
            fail=obj_fall,
        )

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        # 1. rotation reward
        angle = info["rotation_angle"]
        reward = 20 * angle

        # 2. velocity penalty
        obj_vel = info["obj_vel"]
        reward += -0.1 * obj_vel

        # 3. falling penalty
        obj_fall = info["obj_fall"]
        reward += -50.0 * obj_fall

        # 4. effort penalty
        power = torch.abs(info["power"])
        reward += -0.0003 * power

        # 5. torque penalty
        qf = info["qf"]
        qf_norm = torch.linalg.norm(qf, dim=-1)
        reward += -0.0003 * qf_norm

        # 6. finger object distance reward
        obj_tip_dist = info["obj_tip_dist"]
        distance_rew = 0.1 / (0.02 + 4 * obj_tip_dist)
        reward += torch.mean(torch.clip(distance_rew, 0, 1), dim=-1)

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 4.0


@register_env("RotateSingleObjectInHandLevel0-v1", max_episode_steps=300)
class RotateSingleObjectInHandLevel0(RotateSingleObjectInHand):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            robot_init_qpos_noise=0.02,
            obj_init_pos_noise=0.02,
            difficulty_level=0,
            **kwargs,
        )


@register_env("RotateSingleObjectInHandLevel1-v1", max_episode_steps=300)
class RotateSingleObjectInHandLevel1(RotateSingleObjectInHand):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            robot_init_qpos_noise=0.02,
            obj_init_pos_noise=0.02,
            difficulty_level=1,
            **kwargs,
        )


@register_env(
    "RotateSingleObjectInHandLevel2-v1",
    max_episode_steps=300,
    asset_download_ids=["ycb"],
)
class RotateSingleObjectInHandLevel2(RotateSingleObjectInHand):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            robot_init_qpos_noise=0.02,
            obj_init_pos_noise=0.02,
            difficulty_level=2,
            **kwargs,
        )


@register_env(
    "RotateSingleObjectInHandLevel3-v1",
    max_episode_steps=300,
    asset_download_ids=["ycb"],
)
class RotateSingleObjectInHandLevel3(RotateSingleObjectInHand):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            robot_init_qpos_noise=0.02,
            obj_init_pos_noise=0.02,
            difficulty_level=3,
            **kwargs,
        )
