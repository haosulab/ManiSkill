from collections import OrderedDict
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import sapien.core as sapien
from sapien.core import Pose

from mani_skill2 import format_path
from mani_skill2.agents.robots.mobile_panda import DummyMobileAgent
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.utils.common import random_choice
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.sapien_utils import (
    get_actor_state,
    get_articulation_padded_state,
    parse_urdf_config,
    vectorize_pose
)
from mani_skill2.sensors.camera import CameraConfig


class MS1BaseEnv(BaseEnv):
    DEFAULT_MODEL_JSON: str
    ASSET_UID: str
    agent: DummyMobileAgent

    def __init__(
        self,
        *args,
        asset_root: str = "{ASSET_DIR}/partnet_mobility/dataset",
        model_json: str = None,
        model_ids: List[str] = (),
        **kwargs,
    ):
        self.asset_root = Path(format_path(asset_root))

        if model_json is None:
            model_json = self.DEFAULT_MODEL_JSON
        model_json = format_path(model_json)
        self.model_db: Dict[str, Dict] = load_json(model_json)

        if isinstance(model_ids, str):
            model_ids = [model_ids]
        if len(model_ids) == 0:
            model_ids = sorted(self.model_db.keys())
        assert len(model_ids) > 0, model_json

        self.model_ids = model_ids
        self.model_id = None

        self.model_urdf_paths = {}
        for model_id in self.model_ids:
            self.model_urdf_paths[model_id] = self.find_urdf_path(model_id)

        super().__init__(*args, **kwargs)

    def find_urdf_path(self, model_id):
        model_dir = self.asset_root / str(model_id)

        urdf_names = ["mobility_cvx.urdf", "mobility_fixed.urdf"]
        for urdf_name in urdf_names:
            urdf_path = model_dir / urdf_name
            if urdf_path.exists():
                return urdf_path

        raise FileNotFoundError(
            f"No valid URDF is found for {model_id}."
            "Please download Partnet-Mobility (ManiSkill2):"
            "`python -m mani_skill2.utils.download_asset {}`".format(self.ASSET_UID)
        )

    # -------------------------------------------------------------------------- #
    # Reset
    # -------------------------------------------------------------------------- #
    def _get_default_scene_config(self):
        scene_config = super()._get_default_scene_config()
        # Legacy setting
        scene_config.default_dynamic_friction = 0.5
        scene_config.default_static_friction = 0.5
        return scene_config

    def reset(self, seed=None, reconfigure=False, model_id=None):
        self._prev_actor_pose = None
        self.set_episode_rng(seed)
        _reconfigure = self._set_model(model_id)
        reconfigure = _reconfigure or reconfigure
        ret = super().reset(seed=self._episode_seed, reconfigure=reconfigure)
        return ret

    def _set_model(self, model_id):
        """Set the model id. If not provided, choose one randomly."""
        reconfigure = False

        # Model ID
        if model_id is None:
            model_id = random_choice(self.model_ids, self._episode_rng)
        if model_id != self.model_id:
            reconfigure = True
        self.model_id = model_id
        self.model_info = self.model_db[self.model_id]

        return reconfigure

    def _load_actors(self):
        # Create a collision ground plane
        ground = self._add_ground(render=False)
        # Specify a collision (ignore) group to avoid collision with robot torso
        cs = ground.get_collision_shapes()[0]
        cg = cs.get_collision_groups()
        cg[2] = cg[2] | 1 << 30
        cs.set_collision_groups(*cg)

        if self.bg_name is None:
            # Create a visual ground box
            rend_mtl = self._renderer.create_material()
            rend_mtl.base_color = [0.06, 0.08, 0.12, 1]
            rend_mtl.metallic = 0.0
            rend_mtl.roughness = 0.9
            rend_mtl.specular = 0.8
            builder = self._scene.create_actor_builder()
            builder.add_box_visual(
                pose=Pose([0, 0, -1]), half_size=[50, 50, 1], material=rend_mtl
            )
            visual_ground = builder.build_static(name="visual_ground")

    def _load_partnet_mobility(
        self, fix_root_link=True, scale=1.0, urdf_config: dict = None
    ):
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = fix_root_link
        loader.scale = scale
        loader.load_multiple_collisions_from_file = True

        urdf_path = self.model_urdf_paths[self.model_id]
        urdf_config = parse_urdf_config(urdf_config or {}, self._scene)

        articulation = loader.load(str(urdf_path), config=urdf_config)
        return articulation

    def _register_render_cameras(self):
        p = [-1.5, 0, 1.5]
        q = [0.9238795, 0, 0.3826834, 0]
        return CameraConfig("render_camera", p, q, 512, 512, 1, 0.01, 10)

    def _setup_viewer(self):
        super()._setup_viewer()
        self._viewer.set_camera_xyz(-2, 0, 3)
        self._viewer.set_camera_rpy(0, -0.8, 0)

    # -------------------------------------------------------------------------- #
    # Success
    # -------------------------------------------------------------------------- #
    def check_actor_static(self, actor: sapien.Actor, max_v=None, max_ang_v=None):
        """Check whether the actor is static by finite difference.
        Note that the angular velocity is normalized by pi due to legacy issues.
        """
        from mani_skill2.utils.geometry import angle_distance

        pose = actor.get_pose()

        if self._elapsed_steps <= 1:
            flag_v = (max_v is None) or (np.linalg.norm(actor.get_velocity()) <= max_v)
            flag_ang_v = (max_ang_v is None) or (
                np.linalg.norm(actor.get_angular_velocity()) <= max_ang_v
            )
        else:
            dt = 1.0 / self._control_freq
            flag_v = (max_v is None) or (
                np.linalg.norm(pose.p - self._prev_actor_pose.p) <= max_v * dt
            )
            flag_ang_v = (max_ang_v is None) or (
                angle_distance(self._prev_actor_pose, pose) <= max_ang_v * dt
            )

        # CAUTION: carefully deal with it for MPC
        self._prev_actor_pose = pose
        return flag_v and flag_ang_v

    # ---------------------------------------------------------------------------- #
    # Observation
    # ---------------------------------------------------------------------------- #
    def _get_obs_agent(self):
        obs = super()._get_obs_agent()
        if self._obs_mode not in ["state", "state_dict"]:
            obs["base_pose"] = vectorize_pose(self.agent.base_pose)
        return obs
    
    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict()
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(self._get_obs_priviledged())
        return obs

    def _get_obs_priviledged(self):
        obs = OrderedDict()
        actors = self._get_task_actors()
        if len(actors) > 0:
            actors_state = np.hstack([get_actor_state(actor) for actor in actors])
            obs["actors"] = actors_state

        arts_and_dofs = self._get_task_articulations()
        arts_state = np.hstack(
            [
                get_articulation_padded_state(art, max_dof)
                for art, max_dof in arts_and_dofs
            ]
        )
        obs["articulations"] = arts_state

        obs.update(self.agent.get_fingers_info())
        return obs

    def _get_task_actors(self) -> List[sapien.Actor]:
        """Get task-relevant actors (for privileged states)."""
        return []

    def _get_task_articulations(self) -> List[Tuple[sapien.Articulation, int]]:
        """Get task-relevant articulations (for privileged states).
        Each element is (art_obj, max_dof).
        """
        return []
