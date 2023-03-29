from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.euler import euler2quat
from transforms3d.quaternions import axangle2quat, qmult

from mani_skill2 import ASSET_DIR, format_path
from mani_skill2.utils.common import random_choice
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import set_actor_visibility, vectorize_pose

from .base_env import StationaryManipulationEnv


class PickSingleEnv(StationaryManipulationEnv):
    DEFAULT_ASSET_ROOT: str
    DEFAULT_MODEL_JSON: str

    obj: sapien.Actor  # target object

    def __init__(
        self,
        asset_root: str = None,
        model_json: str = None,
        model_ids: List[str] = (),
        obj_init_rot_z=True,
        obj_init_rot=0,
        goal_thresh=0.025,
        **kwargs,
    ):
        if asset_root is None:
            asset_root = self.DEFAULT_ASSET_ROOT
        self.asset_root = Path(format_path(asset_root))

        if model_json is None:
            model_json = self.DEFAULT_MODEL_JSON
        # NOTE(jigu): absolute path will overwrite asset_root
        model_json = self.asset_root / format_path(model_json)

        if not model_json.exists():
            raise FileNotFoundError(
                f"{model_json} is not found."
                "Please download the corresponding assets:"
                "`python -m mani_skill2.utils.download_asset ${ENV_ID}`."
            )
        self.model_db: Dict[str, Dict] = load_json(model_json)

        if isinstance(model_ids, str):
            model_ids = [model_ids]
        if len(model_ids) == 0:
            model_ids = sorted(self.model_db.keys())
        assert len(model_ids) > 0, model_json
        self.model_ids = model_ids

        self.model_id = model_ids[0]
        self.model_scale = None
        self.model_bbox_size = None

        self.obj_init_rot_z = obj_init_rot_z
        self.obj_init_rot = obj_init_rot
        self.goal_thresh = goal_thresh

        self._check_assets()
        super().__init__(**kwargs)

    def _check_assets(self):
        """Check whether the assets exist."""
        pass

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)
        self._load_model()
        self.obj.set_damping(0.1, 0.1)
        self.goal_site = self._build_sphere_site(self.goal_thresh)

    def _load_model(self):
        """Load the target object."""
        raise NotImplementedError

    def reset(self, seed=None, reconfigure=False, model_id=None, model_scale=None):
        self.set_episode_rng(seed)
        _reconfigure = self._set_model(model_id, model_scale)
        reconfigure = _reconfigure or reconfigure
        return super().reset(seed=self._episode_seed, reconfigure=reconfigure)

    def _set_model(self, model_id, model_scale):
        """Set the model id and scale. If not provided, choose one randomly."""
        reconfigure = False

        if model_id is None:
            model_id = random_choice(self.model_ids, self._episode_rng)
        if model_id != self.model_id:
            self.model_id = model_id
            reconfigure = True

        if model_scale is None:
            model_scales = self.model_db[self.model_id].get("scales")
            if model_scales is None:
                model_scale = 1.0
            else:
                model_scale = random_choice(model_scales, self._episode_rng)
        if model_scale != self.model_scale:
            self.model_scale = model_scale
            reconfigure = True

        model_info = self.model_db[self.model_id]
        if "bbox" in model_info:
            bbox = model_info["bbox"]
            bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])
            self.model_bbox_size = bbox_size * self.model_scale
        else:
            self.model_bbox_size = None

        return reconfigure

    def _get_init_z(self):
        return 0.5

    def _settle(self, t):
        sim_steps = int(self.sim_freq * t)
        for _ in range(sim_steps):
            self._scene.step()

    def _initialize_actors(self):
        # The object will fall from a certain height
        xy = self._episode_rng.uniform(-0.1, 0.1, [2])
        z = self._get_init_z()
        p = np.hstack([xy, z])
        q = [1, 0, 0, 0]

        # Rotate along z-axis
        if self.obj_init_rot_z:
            ori = self._episode_rng.uniform(0, 2 * np.pi)
            q = euler2quat(0, 0, ori)

        # Rotate along a random axis by a small angle
        if self.obj_init_rot > 0:
            axis = self._episode_rng.uniform(-1, 1, 3)
            axis = axis / max(np.linalg.norm(axis), 1e-6)
            ori = self._episode_rng.uniform(0, self.obj_init_rot)
            q = qmult(q, axangle2quat(axis, ori, True))
        self.obj.set_pose(Pose(p, q))

        # Move the robot far away to avoid collision
        # The robot should be initialized later
        self.agent.robot.set_pose(Pose([-10, 0, 0]))

        # Lock rotation around x and y
        self.obj.lock_motion(0, 0, 0, 1, 1, 0)
        self._settle(0.5)

        # Unlock motion
        self.obj.lock_motion(0, 0, 0, 0, 0, 0)
        # NOTE(jigu): Explicit set pose to ensure the actor does not sleep
        self.obj.set_pose(self.obj.pose)
        self.obj.set_velocity(np.zeros(3))
        self.obj.set_angular_velocity(np.zeros(3))
        self._settle(0.5)

        # Some objects need longer time to settle
        lin_vel = np.linalg.norm(self.obj.velocity)
        ang_vel = np.linalg.norm(self.obj.angular_velocity)
        if lin_vel > 1e-3 or ang_vel > 1e-2:
            self._settle(0.5)

    @property
    def obj_pose(self):
        """Get the center of mass (COM) pose."""
        return self.obj.pose.transform(self.obj.cmass_local_pose)

    def _initialize_task(self, max_trials=100):
        REGION = [[-0.15, -0.25], [0.15, 0.25]]
        MAX_HEIGHT = 0.5
        MIN_DIST = self.goal_thresh * 2

        # TODO(jigu): Is the goal ambiguous?
        obj_pos = self.obj_pose.p

        # Sample a goal position far enough from the object
        for _ in range(max_trials):
            goal_xy = self._episode_rng.uniform(*REGION)
            goal_z = self._episode_rng.uniform(0, MAX_HEIGHT) + obj_pos[2]
            goal_z = min(goal_z, MAX_HEIGHT)
            goal_pos = np.hstack([goal_xy, goal_z])
            if np.linalg.norm(goal_pos - obj_pos) > MIN_DIST:
                break

        self.goal_pos = goal_pos
        self.goal_site.set_pose(Pose(self.goal_pos))

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.tcp.pose),
            goal_pos=self.goal_pos,
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                tcp_to_goal_pos=self.goal_pos - self.tcp.pose.p,
                obj_pose=vectorize_pose(self.obj_pose),
                tcp_to_obj_pos=self.obj_pose.p - self.tcp.pose.p,
                obj_to_goal_pos=self.goal_pos - self.obj_pose.p,
            )
        return obs

    def check_robot_static(self, thresh=0.2):
        # Assume that the last two DoF is gripper
        qvel = self.agent.robot.get_qvel()[:-2]
        return np.max(np.abs(qvel)) <= thresh

    def evaluate(self, **kwargs):
        obj_to_goal_pos = self.goal_pos - self.obj_pose.p
        is_obj_placed = np.linalg.norm(obj_to_goal_pos) <= self.goal_thresh
        is_robot_static = self.check_robot_static()
        return dict(
            obj_to_goal_pos=obj_to_goal_pos,
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            success=is_obj_placed and is_robot_static,
        )

    def compute_dense_reward(self, info, **kwargs):

        # Sep. 14, 2022:
        # We changed the original complex reward to simple reward,
        # since the original reward can be unfriendly for RL,
        # even though MPC can solve many objects through the original reward.

        reward = 0.0

        if info["success"]:
            reward = 10.0
        else:
            obj_pose = self.obj_pose

            # reaching reward
            tcp_wrt_obj_pose = obj_pose.inv() * self.tcp.pose
            tcp_to_obj_dist = np.linalg.norm(tcp_wrt_obj_pose.p)
            reaching_reward = 1 - np.tanh(
                3.0
                * np.maximum(
                    tcp_to_obj_dist - np.linalg.norm(self.model_bbox_size), 0.0
                )
            )
            reward = reward + reaching_reward

            # grasp reward
            is_grasped = self.agent.check_grasp(self.obj, max_angle=30)
            reward += 3.0 if is_grasped else 0.0

            # reaching-goal reward
            if is_grasped:
                obj_to_goal_pos = self.goal_pos - obj_pose.p
                obj_to_goal_dist = np.linalg.norm(obj_to_goal_pos)
                reaching_goal_reward = 3 * (1 - np.tanh(3.0 * obj_to_goal_dist))
                reward += reaching_goal_reward

        return reward

    def compute_dense_reward_legacy(self, info, **kwargs):
        # original complex reward that is geometry-independent,
        # which ensures that MPC can successfully pick up most objects,
        # but can be unfriendly for RL.

        reward = 0.0
        # hard code gripper info
        finger_length = 0.025
        gripper_width = (
            self.agent.robot.get_qlimits()[-1, 1] * 2
        )  # NOTE: hard-coded with panda

        if info["success"]:
            reward = 10.0
        else:
            obj_pose = self.obj_pose
            # grasp pose rotation reward
            grasp_rot_loss_fxn = lambda A: np.tanh(
                1 / 8 * np.trace(A.T @ A)
            )  # trace(A.T @ A) has range [0,8] for A being difference of rotation matrices
            tcp_pose_wrt_obj = obj_pose.inv() * self.tcp.pose
            tcp_rot_wrt_obj = tcp_pose_wrt_obj.to_transformation_matrix()[:3, :3]
            gt_rot_x_diff_1 = (
                np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]) - tcp_rot_wrt_obj
            )
            gt_rot_x_diff_2 = (
                np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]) - tcp_rot_wrt_obj
            )
            gt_rot_y_diff_1 = (
                np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) - tcp_rot_wrt_obj
            )
            gt_rot_y_diff_2 = (
                np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]) - tcp_rot_wrt_obj
            )

            bbox_x_loss = np.minimum(
                grasp_rot_loss_fxn(gt_rot_x_diff_1), grasp_rot_loss_fxn(gt_rot_x_diff_2)
            )
            bbox_y_loss = np.minimum(
                grasp_rot_loss_fxn(gt_rot_y_diff_1), grasp_rot_loss_fxn(gt_rot_y_diff_2)
            )
            grasp_rot_loss = 1.0
            should_rotate = False
            if self.model_bbox_size[0] < gripper_width:
                should_rotate = True
                grasp_rot_loss = np.minimum(grasp_rot_loss, bbox_x_loss)
            if self.model_bbox_size[1] < gripper_width:
                should_rotate = True
                grasp_rot_loss = np.minimum(grasp_rot_loss, bbox_y_loss)
            rotated_properly = not should_rotate or grasp_rot_loss < 0.1
            reward = reward + (1 - grasp_rot_loss)

            if rotated_properly:
                # reaching reward
                tcp_wrt_obj_pose = obj_pose.inv() * self.tcp.pose
                tcp_to_obj_dist = tcp_wrt_obj_pose.p
                grasp_from_above_height_offset = (
                    self.model_bbox_size[2] / 2 - finger_length
                )  # account for finger length
                if grasp_from_above_height_offset > 0.01:
                    # for objects that are tall, gripper should try to grasp the top of the object
                    tcp_to_obj_dist[2] -= grasp_from_above_height_offset
                else:
                    # for objects that are flat, gripper should try to reach deeper in the object
                    tcp_to_obj_dist[2] += min(self.model_bbox_size[2] / 2, 0.005)
                tcp_to_obj_dist = np.linalg.norm(tcp_to_obj_dist)
                reaching_reward = 1 - np.tanh(4.0 * tcp_to_obj_dist)
                if self.model_bbox_size[2] / 2 > finger_length:
                    reached = tcp_to_obj_dist < 0.01
                else:
                    reached = tcp_to_obj_dist < finger_length + 0.005
                reward += reaching_reward
                if not reached and should_rotate:
                    # encourage grippers to open
                    reward = (
                        reward
                        + (
                            gripper_width
                            - np.sum(
                                gripper_width / 2 - self.agent.robot.get_qpos()[-2:]
                            )
                        )
                        / gripper_width
                    )
                else:
                    reward = reward + 1
                    # grasp reward
                    is_grasped = self.agent.check_grasp(self.obj, max_angle=30)
                    reward += 2.0 if is_grasped else 0.0

                    # reaching-goal reward
                    if is_grasped:
                        obj_to_goal_pos = self.goal_pos - obj_pose.p
                        obj_to_goal_dist = np.linalg.norm(obj_to_goal_pos)
                        reaching_reward2 = 2 * (1 - np.tanh(3 * obj_to_goal_dist))
                        reward += reaching_reward2
            else:
                reward = reward - 15 * np.maximum(
                    obj_pose.p[2]
                    + self.model_bbox_size[2] / 2
                    + 0.02
                    - self.tcp.pose.p[2],
                    0.0,
                )
                reward = reward - 15 * np.linalg.norm(
                    obj_pose.p[:2] - self.tcp.pose.p[:2]
                )
                reward = reward - 15 * np.sum(
                    gripper_width / 2 - self.agent.robot.get_qpos()[-2:]
                )  # ensures that gripper is open

        return reward

    def render(self, mode="human"):
        if mode in ["human", "rgb_array"]:
            set_actor_visibility(self.goal_site, 0.5)
            ret = super().render(mode=mode)
            set_actor_visibility(self.goal_site, 0.0)
        else:
            ret = super().render(mode=mode)
        return ret

    def get_state(self) -> np.ndarray:
        state = super().get_state()
        return np.hstack([state, self.goal_pos])

    def set_state(self, state):
        self.goal_pos = state[-3:]
        super().set_state(state[:-3])


# ---------------------------------------------------------------------------- #
# YCB
# ---------------------------------------------------------------------------- #
def build_actor_ycb(
    model_id: str,
    scene: sapien.Scene,
    scale: float = 1.0,
    physical_material: sapien.PhysicalMaterial = None,
    density=1000,
    root_dir=ASSET_DIR / "mani_skill2_ycb",
):
    builder = scene.create_actor_builder()
    model_dir = Path(root_dir) / "models" / model_id

    collision_file = str(model_dir / "collision.obj")
    builder.add_multiple_collisions_from_file(
        filename=collision_file,
        scale=[scale] * 3,
        material=physical_material,
        density=density,
    )

    visual_file = str(model_dir / "textured.obj")
    builder.add_visual_from_file(filename=visual_file, scale=[scale] * 3)

    actor = builder.build()
    return actor


@register_env("PickSingleYCB-v0", max_episode_steps=200)
class PickSingleYCBEnv(PickSingleEnv):
    DEFAULT_ASSET_ROOT = "{ASSET_DIR}/mani_skill2_ycb"
    DEFAULT_MODEL_JSON = "info_pick_v0.json"

    def _check_assets(self):
        models_dir = self.asset_root / "models"
        for model_id in self.model_ids:
            model_dir = models_dir / model_id
            if not model_dir.exists():
                raise FileNotFoundError(
                    f"{model_dir} is not found."
                    "Please download (ManiSkill2) YCB models:"
                    "`python -m mani_skill2.utils.download_asset ycb`."
                )

            collision_file = model_dir / "collision.obj"
            if not collision_file.exists():
                raise FileNotFoundError(
                    "convex.obj has been renamed to collision.obj. "
                    "Please re-download YCB models."
                )

    def _load_model(self):
        density = self.model_db[self.model_id].get("density", 1000)
        self.obj = build_actor_ycb(
            self.model_id,
            self._scene,
            scale=self.model_scale,
            density=density,
            root_dir=self.asset_root,
        )
        self.obj.name = self.model_id

    def _get_init_z(self):
        bbox_min = self.model_db[self.model_id]["bbox"]["min"]
        return -bbox_min[2] * self.model_scale + 0.05

    def _initialize_agent(self):
        if self.model_bbox_size[2] > 0.15:
            return super()._initialize_agent_v1()
        else:
            return super()._initialize_agent()


# ---------------------------------------------------------------------------- #
# EGAD
# ---------------------------------------------------------------------------- #
def build_actor_egad(
    model_id: str,
    scene: sapien.Scene,
    scale: float = 1.0,
    physical_material: sapien.PhysicalMaterial = None,
    density=100,
    render_material: sapien.RenderMaterial = None,
    root_dir=ASSET_DIR / "mani_skill2_egad",
):
    builder = scene.create_actor_builder()
    # A heuristic way to infer split
    split = "train" if "_" in model_id else "eval"

    collision_file = Path(root_dir) / f"egad_{split}_set_coacd" / f"{model_id}.obj"
    builder.add_multiple_collisions_from_file(
        filename=str(collision_file),
        scale=[scale] * 3,
        material=physical_material,
        density=density,
    )

    visual_file = Path(root_dir) / f"egad_{split}_set" / f"{model_id}.obj"
    builder.add_visual_from_file(
        filename=str(visual_file), scale=[scale] * 3, material=render_material
    )

    actor = builder.build()
    return actor


@register_env("PickSingleEGAD-v0", max_episode_steps=200, obj_init_rot=0.2)
class PickSingleEGADEnv(PickSingleEnv):
    DEFAULT_ASSET_ROOT = "{ASSET_DIR}/mani_skill2_egad"
    DEFAULT_MODEL_JSON = "info_pick_train_v0.json"

    def _check_assets(self):
        splits = set()
        for model_id in self.model_ids:
            split = "train" if "_" in model_id else "eval"
            splits.add(split)

        for split in splits:
            collision_dir = self.asset_root / f"egad_{split}_set_coacd"
            visual_dir = self.asset_root / f"egad_{split}_set"
            if not (collision_dir.exists() and visual_dir.exists()):
                raise FileNotFoundError(
                    f"{collision_dir} or {visual_dir} is not found. "
                    "Please download (ManiSkill2) EGAD models:"
                    "`python -m mani_skill2.utils.download_asset egad`."
                )

    def _load_model(self):
        mat = self._renderer.create_material()
        color = self._episode_rng.uniform(0.2, 0.8, 3)
        color = np.hstack([color, 1.0])
        mat.set_base_color(color)
        mat.metallic = 0.0
        mat.roughness = 0.1

        self.obj = build_actor_egad(
            self.model_id,
            self._scene,
            scale=self.model_scale,
            render_material=mat,
            density=100,
            root_dir=self.asset_root,
        )
        self.obj.name = self.model_id

    def _get_init_z(self):
        bbox_min = self.model_db[self.model_id]["bbox"]["min"]
        return -bbox_min[2] * self.model_scale + 0.05

    def _initialize_actors(self):
        super()._initialize_actors()

        # Some objects need longer time to settle
        lin_vel = np.linalg.norm(self.obj.velocity)
        ang_vel = np.linalg.norm(self.obj.angular_velocity)
        if lin_vel > 1e-3 or ang_vel > 1e-2:
            self._settle(0.5)
