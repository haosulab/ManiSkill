from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np
import sapien
import sapien.physx as physx
import sapien.render
import torch
from transforms3d.euler import euler2quat
from transforms3d.quaternions import axangle2quat, qmult

from mani_skill2 import ASSET_DIR, format_path
from mani_skill2.utils.common import random_choice
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import to_tensor
from mani_skill2.utils.scene_builder import TableSceneBuilder
from mani_skill2.utils.structs.actor import Actor
from mani_skill2.utils.structs.pose import Pose, vectorize_pose

from .base_env import StationaryManipulationEnv


class PickSingleEnv(StationaryManipulationEnv):
    DEFAULT_ASSET_ROOT: str
    DEFAULT_MODEL_JSON: str

    obj: Actor  # target object

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
        self.models_loaded = False
        super().__init__(**kwargs)

    def _check_assets(self):
        """Check whether the assets exist."""
        pass

    def _load_actors(self):
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        self._load_model()
        self.obj.set_linear_damping(0.1)
        self.obj.set_angular_damping(0.1)
        self.goal_site = self._build_sphere_site(self.goal_thresh)

    def _load_model(self):
        """Load the target object."""
        raise NotImplementedError

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        self._set_episode_rng(seed)
        model_scale = options.pop("model_scale", None)
        model_id = options.pop("model_id", None)
        reconfigure = options.pop("reconfigure", False)
        if not self.models_loaded or not physx.is_gpu_enabled():
            # we can only reconfigure once with GPU enabled.
            _reconfigure = self._set_model(model_id, model_scale)
            reconfigure = _reconfigure or reconfigure
            options["reconfigure"] = reconfigure
            self.models_loaded = True
        return super().reset(seed=self._episode_seed, options=options)

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
            self.model_bbox_size = to_tensor(bbox_size * self.model_scale)
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
        self.table_scene.initialize()
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

        self.obj.set_pose(Pose.create_from_pq(p, q))

        # Move the robot far away to avoid collision
        # The robot should be initialized later
        # # self.agent.robot.set_pose(Pose([-10, 0, 0]))

        # # Lock rotation around x and y
        # obj_comp.set_locked_motion_axes([0, 0, 0, 1, 1, 0])
        # self._settle(0.5)

        # # Unlock motion
        # obj_comp.set_locked_motion_axes([0, 0, 0, 0, 0, 0])
        # # NOTE(jigu): Explicit set pose to ensure the entity does not sleep
        # obj_comp.set_pose(obj_comp.pose)
        # obj_comp.set_linear_velocity(np.zeros(3))
        # obj_comp.set_angular_velocity(np.zeros(3))
        # # self._settle(0.5)

        # # Some objects need longer time to settle
        # lin_vel = np.linalg.norm(obj_comp.linear_velocity)
        # ang_vel = np.linalg.norm(obj_comp.angular_velocity)
        # if lin_vel > 1e-3 or ang_vel > 1e-2:
        #     self._settle(0.5)

    @property
    def obj_pose(self):
        """Get the center of mass (COM) pose."""
        return (
            self.obj.pose
            # * self.obj.find_component_by_type(
            #     physx.PhysxRigidDynamicComponent
            # ).cmass_local_pose
        )

    def _initialize_task(self, max_trials=100):
        REGION = [[-0.15, -0.25], [0.15, 0.25]]
        MAX_HEIGHT = 0.5
        MIN_DIST = self.goal_thresh * 2

        # TODO(jigu): Is the goal ambiguous?
        obj_pos = self.obj_pose.p

        # Sample a goal position far enough from the object
        goal_poss = []
        for i in range(len(obj_pos)):
            for _ in range(max_trials):
                goal_xy = to_tensor(self._episode_rng.uniform(*REGION))
                goal_z = self._episode_rng.uniform(0, MAX_HEIGHT) + obj_pos[i, 2]
                goal_z = min(goal_z, MAX_HEIGHT)
                goal_pos = torch.hstack([goal_xy, goal_z])
                if torch.linalg.norm(goal_pos - obj_pos) > MIN_DIST:
                    goal_poss.append(goal_pos)
                    break
        self.goal_pos = torch.vstack(goal_poss)
        self.goal_site.set_pose(Pose.create_from_pq(p=self.goal_pos))

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.agent.tcp.pose),
            goal_pos=self.goal_pos,
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                tcp_to_goal_pos=self.goal_pos - self.agent.tcp.pose.p,
                obj_pose=vectorize_pose(self.obj_pose),
                tcp_to_obj_pos=self.obj_pose.p - self.agent.tcp.pose.p,
                obj_to_goal_pos=self.goal_pos - self.obj_pose.p,
            )
        return obs

    def check_robot_static(self, thresh=0.2):
        # Assume that the last two DoF is gripper
        qvel = self.agent.robot.get_qvel()[..., :-2]
        return torch.max(torch.abs(qvel), 1)[0] <= thresh

    def evaluate(self, obs):
        obj_to_goal_pos = self.goal_pos - self.obj_pose.p
        is_obj_placed = torch.linalg.norm(obj_to_goal_pos, axis=1) <= self.goal_thresh
        is_robot_static = self.check_robot_static()
        return dict(
            obj_to_goal_pos=obj_to_goal_pos,
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            success=torch.logical_and(is_obj_placed, is_robot_static)
        )

    def compute_dense_reward(self, obs, action, info):
        # Sep. 14, 2022:
        # We changed the original complex reward to simple reward,
        # since the original reward can be unfriendly for RL,
        # even though MPC can solve many objects through the original reward.


        obj_pose = self.obj_pose

        # reaching reward
        tcp_wrt_obj_pose: Pose = obj_pose.inv() * self.agent.tcp.pose
        tcp_to_obj_dist = torch.linalg.norm(tcp_wrt_obj_pose.p, axis=1)
        reaching_reward = 1 - torch.tanh(
            3.0
            * torch.maximum(
                tcp_to_obj_dist - torch.linalg.norm(self.model_bbox_size), torch.tensor(0.0)
            )
        )
        reward = reaching_reward

        # grasp reward
        is_grasped = self.agent.is_grasping(self.obj, max_angle=30)
        reward += 3 * is_grasped

        # reaching-goal reward
        obj_to_goal_pos = self.goal_pos - obj_pose.p
        obj_to_goal_dist = torch.linalg.norm(obj_to_goal_pos, axis=1)
        reaching_goal_reward = 3 * (1 - torch.tanh(3.0 * obj_to_goal_dist))
        reward += reaching_goal_reward * is_grasped

        reward[info["success"]] = 10.0
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 10.0

    def get_state(self):
        state = super().get_state()
        return torch.hstack([state, self.goal_pos])

    def set_state(self, state):
        self.goal_pos = state[-3:]
        super().set_state(state[:-3])


# ---------------------------------------------------------------------------- #
# YCB
# ---------------------------------------------------------------------------- #
def build_actor_ycb(
    model_id: str,
    scene: sapien.Scene,
    name: str,
    scale: float = 1.0,
    physical_material: physx.PhysxMaterial = None,
    density=1000,
    root_dir=ASSET_DIR / "mani_skill2_ycb",
):
    builder = scene.create_actor_builder()
    model_dir = Path(root_dir) / "models" / model_id

    collision_file = str(model_dir / "collision.ply")
    builder.add_multiple_convex_collisions_from_file(
        filename=collision_file,
        scale=[scale] * 3,
        material=physical_material,
        density=density,
    )

    visual_file = str(model_dir / "textured.obj")
    builder.add_visual_from_file(filename=visual_file, scale=[scale] * 3)

    actor = builder.build(name=name)
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

            collision_file = model_dir / "collision.ply"
            if not collision_file.exists():
                raise FileNotFoundError(
                    "convex.obj and collision.obj has been renamed to collision.ply. "
                    "Please re-download YCB models."
                )

    def _load_model(self):
        density = self.model_db[self.model_id].get("density", 1000)
        self.obj = build_actor_ycb(
            self.model_id,
            self._scene,
            name=self.model_id,
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
