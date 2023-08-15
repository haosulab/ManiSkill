from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np
import sapien.core as sapien
from sapien.core import Pose

from mani_skill2 import format_path
from mani_skill2.utils.common import random_choice
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import look_at, set_actor_visibility, vectorize_pose

from .base_env import StationaryManipulationEnv
from .pick_single import PickSingleYCBEnv, build_actor_ycb


class PickClutterEnv(StationaryManipulationEnv):
    DEFAULT_EPISODE_JSON: str
    DEFAULT_ASSET_ROOT: str
    DEFAULT_MODEL_JSON: str

    obj: sapien.Actor  # target object

    def __init__(
        self,
        episode_json: str = None,
        asset_root: str = None,
        model_json: str = None,
        **kwargs,
    ):
        # Load episode configurations
        if episode_json is None:
            episode_json = self.DEFAULT_EPISODE_JSON
        episode_json = format_path(episode_json)
        if not Path(episode_json).exists():
            raise FileNotFoundError(
                f"Episode json ({episode_json}) is not found."
                "To download default json:"
                "`python -m mani_skill2.utils.download_asset pick_clutter_ycb`."
            )
        self.episodes: List[Dict] = load_json(episode_json)

        # Root directory of object models
        if asset_root is None:
            asset_root = self.DEFAULT_ASSET_ROOT
        self.asset_root = Path(format_path(asset_root))

        # Information of object models
        if model_json is None:
            model_json = self.DEFAULT_MODEL_JSON
        model_json = self.asset_root / format_path(model_json)
        self.model_db: Dict[str, Dict] = load_json(model_json)

        self.episode_idx = -1

        self.goal_thresh = 0.025

        super().__init__(**kwargs)

    def _load_actors(self):
        self._add_ground(render=self.bg_name is None)

        self.objs: List[sapien.Actor] = []
        self.bbox_sizes = []
        for actor_cfg in self.episode["actors"]:
            model_id = actor_cfg["model_id"]
            model_scale = actor_cfg["scale"]
            obj = self._load_model(model_id, model_scale=model_scale)
            self.objs.append(obj)

            bbox = self.model_db[model_id]["bbox"]
            bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])
            self.bbox_sizes.append(bbox_size * model_scale)

        self.target_site = self._build_sphere_site(
            0.01, color=(1, 1, 0), name="_target_site"
        )
        self.goal_site = self._build_sphere_site(
            0.01, color=(0, 1, 0), name="_goal_site"
        )

    def _load_model(self, model_id, model_scale=1.0) -> sapien.Actor:
        raise NotImplementedError

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        self.set_episode_rng(seed)
        episode_idx = options.pop("episode_idx", None)
        reconfigure = options.pop("reconfigure", False)
        _reconfigure = self._set_episode(episode_idx)
        reconfigure = _reconfigure or reconfigure
        options["reconfigure"] = reconfigure
        return super().reset(seed=self._episode_seed, options=options)

    def _set_episode(self, episode_idx=None):
        reconfigure = False

        if episode_idx is None:
            episode_idx = self._episode_rng.randint(len(self.episodes))
            # episode_idx = (self.episode_idx + 1) % len(self.episodes)

        # TODO(jigu): check whether assets (and scales) are the same instead
        if self.episode_idx != episode_idx:
            reconfigure = True

        self.episode_idx = episode_idx
        self.episode = self.episodes[self.episode_idx]

        return reconfigure

    def _initialize_actors(self):
        for i, actor_cfg in enumerate(self.episode["actors"]):
            pose = np.float32(actor_cfg["pose"])
            # Add a small offset to avoid numerical issues for simulation
            self.objs[i].set_pose(Pose(pose[:3] + [0, 0, 1e-3], pose[3:]))

        # Settle
        self.agent.robot.set_pose(Pose([10, 0, 0]))
        for _ in range(self.control_freq):
            self._scene.step()

    def _initialize_agent(self):
        if self.robot_uid == "panda":
            # fmt: off
            qpos = np.array(
                [0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04]
            )
            # fmt: on
            qpos[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose([-0.544, 0, 0]))
        else:
            raise NotImplementedError(self.robot_uid)

    @property
    def obj_pose(self):
        """Get the center of mass (COM) pose."""
        return self.obj.pose.transform(self.obj.cmass_local_pose)

    def _initialize_task(self):
        self._set_target()
        self._set_goal()

    def _set_target(self):
        visible_inds = []
        for i, actor_cfg in enumerate(self.episode["actors"]):
            if actor_cfg["rep_pts"] is not None:
                visible_inds.append(i)
        assert len(visible_inds) > 0, self.episode_idx

        actor_idx = random_choice(visible_inds, self._episode_rng)
        self.obj = self.objs[actor_idx]
        obj_rep_pts = self.episode["actors"][actor_idx]["rep_pts"]
        self.obj_start_pos = np.float32(random_choice(obj_rep_pts, self._episode_rng))
        self.bbox_size = self.bbox_sizes[actor_idx]

        self.target_site.set_pose(Pose(self.obj_start_pos))

    def _set_goal(self):
        goal_pos = self._episode_rng.uniform([-0.15, -0.25, 0.35], [0.15, 0.25, 0.45])
        self.goal_pos = goal_pos
        self.goal_site.set_pose(Pose(self.goal_pos))

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.tcp.pose),
            goal_pos=self.goal_pos,
            obj_start_pos=self.obj_start_pos,
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
        reward = 0.0

        if info["success"]:
            reward = 10.0
        else:
            obj_pose = self.obj_pose

            # reaching reward
            tcp_wrt_obj_pose = obj_pose.inv() * self.tcp.pose
            tcp_to_obj_dist = np.linalg.norm(tcp_wrt_obj_pose.p)
            reaching_reward = 1 - np.tanh(
                3.0 * np.maximum(tcp_to_obj_dist - np.linalg.norm(self.bbox_size), 0.0)
            )
            reward = reward + reaching_reward

            # grasp reward
            is_grasped = self.agent.check_grasp(self.obj, max_angle=60)
            reward += 3.0 if is_grasped else 0.0

            # reaching-goal reward
            if is_grasped:
                obj_to_goal_pos = self.goal_pos - obj_pose.p
                obj_to_goal_dist = np.linalg.norm(obj_to_goal_pos)
                reaching_goal_reward = 3 * (1 - np.tanh(3.0 * obj_to_goal_dist))
                reward += reaching_goal_reward

        return reward

    def compute_normalized_dense_reward(self, **kwargs):
        return self.compute_dense_reward(**kwargs) / 10.0

    def _register_render_cameras(self):
        cam_cfg = super()._register_render_cameras()
        cam_cfg.pose = look_at([0.3, 0, 1.0], [0.0, 0.0, 0.5])
        return cam_cfg

    def render_human(self):
        set_actor_visibility(self.target_site, 0.8)
        set_actor_visibility(self.goal_site, 0.5)
        ret = super().render_human()
        set_actor_visibility(self.target_site, 0)
        set_actor_visibility(self.goal_site, 0)
        return ret

    def render_rgb_array(self):
        set_actor_visibility(self.target_site, 0.8)
        set_actor_visibility(self.goal_site, 0.5)
        ret = super().render_rgb_array()
        set_actor_visibility(self.target_site, 0)
        set_actor_visibility(self.goal_site, 0)
        return ret

    def get_state(self) -> np.ndarray:
        state = super().get_state()
        # TODO(jigu): whether to add obj_start_pos
        return np.hstack([state, self.goal_pos])

    def set_state(self, state):
        self.goal_pos = state[-3:]
        super().set_state(state[:-3])


@register_env("PickClutterYCB-v0", max_episode_steps=200)
class PickClutterYCBEnv(PickClutterEnv):
    DEFAULT_EPISODE_JSON = "{ASSET_DIR}/pick_clutter/ycb_train_5k.json.gz"
    DEFAULT_ASSET_ROOT = PickSingleYCBEnv.DEFAULT_ASSET_ROOT
    DEFAULT_MODEL_JSON = PickSingleYCBEnv.DEFAULT_MODEL_JSON

    def _load_model(self, model_id, model_scale=1.0):
        density = self.model_db[model_id].get("density", 1000)
        obj = build_actor_ycb(
            model_id,
            self._scene,
            scale=model_scale,
            density=density,
            root_dir=self.asset_root,
        )
        obj.name = model_id
        obj.set_damping(0.1, 0.1)
        return obj
