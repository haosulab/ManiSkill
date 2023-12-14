from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import sapien
import sapien.physx as physx
from sapien import Pose

from mani_skill2 import format_path
from mani_skill2.agents.robots.mobile_panda import DummyMobileAgent
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.building.ground import build_tesselated_square_floor
from mani_skill2.utils.common import random_choice
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.sapien_utils import (
    apply_urdf_config,
    get_actor_state,
    get_articulation_padded_state,
    parse_urdf_config,
    vectorize_pose,
)


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
        # scene_config.default_dynamic_friction = 0.5
        # scene_config.default_static_friction = 0.5
        return scene_config

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        self._prev_actor_pose = None
        self._set_episode_rng(seed)
        model_id = options.pop("model_id", None)
        reconfigure = options.pop("reconfigure", False)
        _reconfigure = self._set_model(model_id)
        reconfigure = _reconfigure or reconfigure
        options["reconfigure"] = reconfigure
        return super().reset(seed=self._episode_seed, options=options)

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
        ground = build_tesselated_square_floor(self._scene)
        # TODO (stao): This is quite hacky. Future we expect the robot to be an actual well defined robot without needing to intersect the ground. We should probably deprecate the old ms1 envs eventually
        # Specify a collision (ignore) group to avoid collision with robot torso
        cs = ground.find_component_by_type(
            physx.PhysxRigidStaticComponent
        ).get_collision_shapes()[0]
        cg = cs.get_collision_groups()
        cg[2] = cg[2] | 1 << 30
        cs.set_collision_groups(cg)

    def _load_partnet_mobility(
        self, fix_root_link=True, scale=1.0, urdf_config: dict = None
    ):
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = fix_root_link
        loader.scale = scale
        loader.load_multiple_collisions_from_file = True

        urdf_path = self.model_urdf_paths[self.model_id]
        urdf_config = parse_urdf_config(urdf_config or {}, self._scene)
        apply_urdf_config(loader, urdf_config)
        articulation: physx.PhysxArticulation = loader.load(str(urdf_path))
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
    def check_link_static(
        self, link: physx.PhysxArticulationLinkComponent, max_v=None, max_ang_v=None
    ):
        """Check whether the link is static by finite difference.
        Note that the angular velocity is normalized by pi due to legacy issues.
        """
        from mani_skill2.utils.geometry import angle_distance

        pose = link.get_pose()

        if self._elapsed_steps <= 1:
            flag_v = (max_v is None) or (
                np.linalg.norm(link.get_linear_velocity()) <= max_v
            )
            flag_ang_v = (max_ang_v is None) or (
                np.linalg.norm(link.get_angular_velocity()) <= max_ang_v
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

    def _get_task_actors(self) -> List[sapien.Entity]:
        """Get task-relevant actors (for privileged states)."""
        return []

    def _get_task_articulations(self) -> List[Tuple[physx.PhysxArticulation, int]]:
        """Get task-relevant articulations (for privileged states).
        Each element is (art_obj, max_dof).
        """
        return []
