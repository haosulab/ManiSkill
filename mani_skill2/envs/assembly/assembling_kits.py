from pathlib import Path
from typing import List

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat, quat2euler

from mani_skill2 import format_path
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import look_at, vectorize_pose

from .base_env import StationaryManipulationEnv


@register_env("AssemblingKits-v0", max_episode_steps=200)
class AssemblingKitsEnv(StationaryManipulationEnv):
    def __init__(self, asset_root="{ASSET_DIR}/assembling_kits", **kwargs):
        self.asset_root = Path(format_path(asset_root))

        self._kit_dir = self.asset_root / "kits"
        self._models_dir = self.asset_root / "models"
        if not (self._kit_dir.exists() and self._models_dir.exists()):
            raise FileNotFoundError(
                "The objects/kits are not found."
                "Please download (ManiSkill2) AssemblingKits assets:"
                "`python -m mani_skill2.utils.download_asset assembling_kits`."
            )

        self._episode_json = load_json(self.asset_root / "episodes.json")
        self._episodes = self._episode_json["episodes"]
        self.episode_idx = None

        self.symmetry = self._episode_json["config"]["symmetry"]
        self.color = self._episode_json["config"]["color"]
        self.object_scale = self._episode_json["config"]["object_scale"]

        super().__init__(**kwargs)

    def reset(self, seed=None, episode_idx=None, reconfigure=False, **kwargs):
        self.set_episode_rng(seed)

        if episode_idx is None:
            episode_idx = self._episode_rng.randint(len(self._episodes))
        if self.episode_idx != episode_idx:
            reconfigure = True
        self.episode_idx = episode_idx

        episode = self._episodes[episode_idx]
        self.kit_id: int = episode["kit"]
        self.spawn_pos = np.float32(episode["spawn_pos"])
        self.object_id: int = episode["obj_to_place"]
        self._other_objects_id: List[int] = episode["obj_in_place"]

        return super().reset(seed=self._episode_seed, reconfigure=reconfigure, **kwargs)

    def _parse_json(self, path):
        """Parse kit JSON information"""
        kit_json = load_json(path)

        self._objects_id = [o["object_id"] for o in kit_json["objects"]]
        self.objects_pos = {
            o["object_id"]: np.float32(o["pos"]) for o in kit_json["objects"]
        }
        # z-axis orientation
        self.objects_rot = {o["object_id"]: o["rot"] for o in kit_json["objects"]}

    def _load_kit(self):
        self._parse_json(self._kit_dir / f"{self.kit_id}.json")

        builder = self._scene.create_actor_builder()
        kit_path = str(self._kit_dir / f"{self.kit_id}.obj")
        builder.add_nonconvex_collision_from_file(kit_path)

        material = self._renderer.create_material()
        material.set_base_color([0.27807487, 0.20855615, 0.16934046, 1.0])
        material.metallic = 0.0
        material.roughness = 0.5
        material.specular = 0.0
        builder.add_visual_from_file(kit_path, material=material)

        return builder.build_static(name="kit")

    def _load_object(self, object_id):
        collision_path = self._models_dir / "collision" / f"{object_id:02d}.obj"
        visual_path = self._models_dir / "visual" / f"{object_id:02d}.obj"

        builder = self._scene.create_actor_builder()
        builder.add_multiple_collisions_from_file(
            str(collision_path), scale=self.object_scale
        )

        material = self._renderer.create_material()
        material.set_base_color(self.color[self._episode_rng.choice(len(self.color))])
        material.metallic = 0.0
        material.roughness = 0.1
        material.specular = 0.0
        builder.add_visual_from_file(
            str(visual_path), scale=self.object_scale, material=material
        )

        return builder.build(f"obj_{object_id:02d}")

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)

        self.kit = self._load_kit()
        self.obj = self._load_object(self.object_id)
        self._other_objs = [self._load_object(i) for i in self._other_objects_id]

    def _initialize_actors(self):
        self.kit.set_pose(sapien.Pose([0, 0, 0.01]))
        self.obj.set_pose(
            sapien.Pose(
                self.spawn_pos,
                euler2quat(0, 0, self._episode_rng.rand() * np.pi * 2),
            )
        )

        for i, o in enumerate(self._other_objs):
            obj_id = self._other_objects_id[i]
            o.set_pose(
                sapien.Pose(
                    np.array(self.objects_pos[obj_id]),
                    euler2quat(0, 0, self.objects_rot[obj_id]),
                )
            )

    def _initialize_task(self):
        self._obj_init_pos = np.float32(self.spawn_pos)
        self._obj_goal_pos = np.float32(self.objects_pos[self.object_id])

    def _get_obs_extra(self):
        obs = dict(
            tcp_pose=vectorize_pose(self.tcp.pose),
            obj_init_pos=self._obj_init_pos,
            obj_goal_pos=self._obj_goal_pos,
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                obj_pose=vectorize_pose(self.obj.pose),
                tcp_to_obj_pos=self.obj.pose.p - self.tcp.pose.p,
                obj_to_goal_pos=self.objects_pos[self.object_id] - self.obj.pose.p,
            )
        return obs

    def _check_pos_diff(self, pos_eps=2e-2):
        pos_diff = self.objects_pos[self.object_id][:2] - self.obj.get_pose().p[:2]
        pos_diff_norm = np.linalg.norm(pos_diff)
        return pos_diff, pos_diff_norm, pos_diff_norm < pos_eps

    def _check_rot_diff(self, rot_eps=np.deg2rad(4)):
        rot = quat2euler(self.obj.get_pose().q)[-1]  # Check z-axis rotation
        rot_diff = 0
        if self.symmetry[self.object_id] > 0:
            rot_diff = (
                np.abs(rot - self.objects_rot[self.object_id])
                % self.symmetry[self.object_id]
            )
            if rot_diff > (self.symmetry[self.object_id] / 2):
                rot_diff = self.symmetry[self.object_id] - rot_diff
        return rot_diff, rot_diff < rot_eps

    def _check_in_slot(self, obj: sapien.Actor, height_eps=3e-3):
        return obj.pose.p[2] < height_eps

    def evaluate(self, **kwargs) -> dict:
        pos_diff, pos_diff_norm, pos_correct = self._check_pos_diff()
        rot_diff, rot_correct = self._check_rot_diff()
        in_slot = self._check_in_slot(self.obj)
        return {
            "pos_diff": pos_diff,
            "pos_diff_norm": pos_diff_norm,
            "pos_correct": pos_correct,
            "rot_diff": rot_diff,
            "rot_correct": rot_correct,
            "in_slot": in_slot,
            "success": pos_correct and rot_correct and in_slot,
        }

    def compute_dense_reward(self, info, **kwargs):
        if info["success"]:
            return 10.0

        reward = 0.0
        gripper_width = (
            self.agent.robot.get_qlimits()[-1, 1] * 2
        )  # NOTE: hard-coded with panda

        # reaching reward
        gripper_to_obj_dist = np.linalg.norm(self.tcp.pose.p - self.obj.pose.p)
        reaching_reward = 1 - np.tanh(4.0 * np.maximum(gripper_to_obj_dist - 0.01, 0.0))
        reward += reaching_reward

        # object position and rotation reward
        pos_diff_norm_reward = 1 - np.tanh(4.0 * info["pos_diff_norm"])
        rot_diff_reward = 1 - np.tanh(0.4 * info["rot_diff"])
        rot_diff_reward += 1.0 - np.tanh(
            1.0 - self.obj.pose.to_transformation_matrix()[2, 2]
        )  # encourage object to be parallel to xy-plane
        object_well_positioned = info["pos_correct"] and info["rot_correct"]
        if object_well_positioned:
            reward += 1.0

        # grasp reward
        is_grasped = self.agent.check_grasp(
            self.obj, max_angle=30
        )  # max_angle ensures that the gripper grasps the object appropriately, not in a strange pose
        if is_grasped or object_well_positioned:
            reward += 2.0
            reward += pos_diff_norm_reward
            reward += rot_diff_reward
            if object_well_positioned and rot_diff_reward > 0.98:
                # ungrasp the object
                if not is_grasped:
                    reward += 1.0
                else:
                    reward = (
                        reward
                        + 1.0 * np.sum(self.agent.robot.get_qpos()[-2:]) / gripper_width
                    )

        return reward

    def _register_cameras(self):
        cam_cfg = super()._register_cameras()
        cam_cfg.pose = look_at([0.2, 0, 0.4], [0, 0, 0])
        return cam_cfg

    def _register_render_cameras(self):
        cam_cfg = super()._register_render_cameras()
        cam_cfg.pose = look_at([0.3, 0.3, 0.8], [0.0, 0.0, 0.1])
        return cam_cfg
