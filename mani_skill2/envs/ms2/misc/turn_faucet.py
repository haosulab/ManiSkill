from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import numpy as np
import sapien
import sapien.physx as physx
import sapien.render
import trimesh
import trimesh.sample
from sapien import Pose
from scipy.spatial.distance import cdist
from transforms3d.euler import euler2quat

from mani_skill2 import format_path
from mani_skill2.agents.robots.panda.panda import Panda
from mani_skill2.envs.sapien_env import BaseEnv
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.building.ground import build_ground
from mani_skill2.utils.common import np_random, random_choice
from mani_skill2.utils.geometry import transform_points
from mani_skill2.utils.geometry.trimesh_utils import get_component_mesh
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import (
    get_obj_by_name,
    hex2rgba,
    look_at,
    set_articulation_render_material,
)
from mani_skill2.utils.structs.pose import vectorize_pose


class TurnFaucetBaseEnv(BaseEnv):
    agent: Panda

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_actors(self):
        # builder = self._scene.create_actor_builder()
        # model_dir = Path(osp.dirname(__file__)) / "assets"
        # scale = 1
        # collision_file = str(model_dir / "Sink_19.glb")  # a metal table
        # sink_pose = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0))
        # builder.add_nonconvex_collision_from_file(
        #     filename=collision_file, scale=[scale] * 3, material=None, pose=sink_pose
        # )
        # visual_file = str(model_dir / "Sink_19.glb")
        # builder.add_visual_from_file(
        #     filename=visual_file, scale=[scale] * 3, pose=sink_pose
        # )
        # self.sink = builder.build_static(name="sink")
        # aabb = self.sink.find_component_by_type(
        #     sapien.render.RenderBodyComponent
        # ).compute_global_aabb_tight()
        # sink_height = aabb[1, 2] - aabb[0, 2]

        # self.sink.set_pose(
        #     Pose(p=[-0.24, 0, -sink_height], q=euler2quat(0, 0, -np.pi / 2))
        # )

        build_ground(self._scene)

        # # add wall
        # wall_mtl = sapien.render.RenderMaterial(
        #     base_color=[32 / 255, 67 / 255, 80 / 255, 1],
        #     metallic=0,
        #     roughness=0.9,
        #     specular=0.8,
        # )
        # wall = self._scene.create_actor_builder()
        # half_size = (0.02, 6, 2.1)
        # wall.add_box_collision(half_size=half_size)
        # wall.add_box_visual(half_size=half_size, material=wall_mtl)
        # self.wall = wall.build_static("wall")
        # self.wall.set_pose(Pose(p=[0.25, 0, 1]))

    def _initialize_agent(self):
        if self.robot_uids == "panda":
            # fmt: off
            qpos = np.array([0, -0.785, 0, -2.356, 0, 1.57, 0.785, 0, 0])
            # fmt: on
            qpos[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose([-0.56, 0, 0]))
        else:
            raise NotImplementedError(self.robot_uids)

    def _get_obs_agent(self):
        obs = self.agent.get_proprioception()
        obs["base_pose"] = vectorize_pose(self.agent.robot.pose)
        return obs

    def _register_sensors(self):
        pose = look_at([-0.4, 0, 0.3], [0, 0, 0.1])
        return CameraConfig(
            "base_camera", pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 10
        )

    def _register_human_render_cameras(self):
        pose = look_at([-1.3, 0.6, 0.6], [0.0, 0.0, 0.4])
        return CameraConfig("render_camera", pose.p, pose.q, 512, 512, 1, 0.01, 10)

    def _setup_viewer(self):
        super()._setup_viewer()
        self._viewer.set_camera_pose(look_at([-1.3, 0.6, 0.6], [0.0, 0.0, 0.4]))


@register_env("TurnFaucet-v0", max_episode_steps=200)
class TurnFaucetEnv(TurnFaucetBaseEnv):
    target_link: physx.PhysxArticulationLinkComponent
    target_joint: physx.PhysxJointComponent

    def __init__(
        self,
        asset_root: str = "{ASSET_DIR}/partnet_mobility/dataset",
        model_json: str = "{PACKAGE_ASSET_DIR}/partnet_mobility/meta/info_faucet_train.json",
        model_ids: List[str] = (),
        **kwargs,
    ):
        self.asset_root = Path(format_path(asset_root))

        model_json = format_path(model_json)
        self.model_db: Dict[str, Dict] = load_json(model_json)

        if isinstance(model_ids, str):
            model_ids = [model_ids]
        if len(model_ids) == 0:
            model_ids = sorted(self.model_db.keys())
        assert len(model_ids) > 0, model_json

        # model_ids = list(map(str, model_ids))
        self.model_ids = model_ids
        self.model_id = None
        self.model_scale = None

        # Find and check urdf paths
        self.model_urdf_paths = {}
        for model_id in self.model_ids:
            self.model_urdf_paths[model_id] = self.find_urdf_path(model_id)

        super().__init__(**kwargs)

    def find_urdf_path(self, model_id):
        model_dir = self.asset_root / str(model_id)

        urdf_names = ["mobility_cvx.urdf"]
        for urdf_name in urdf_names:
            urdf_path = model_dir / urdf_name
            if urdf_path.exists():
                return urdf_path

        raise FileNotFoundError(
            f"No valid URDF is found for {model_id}."
            "Please download Partnet-Mobility (ManiSkill2):"
            "`python -m mani_skill2.utils.download_asset partnet_mobility_faucet`."
        )

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        self._set_episode_rng(seed)
        model_id = options.pop("model_id", None)
        model_scale = options.pop("model_scale", None)
        reconfigure = options.pop("reconfigure", False)

        _reconfigure = self._set_model(model_id, model_scale)
        reconfigure = _reconfigure or reconfigure
        options["reconfigure"] = reconfigure
        return super().reset(seed=self._episode_seed, options=options)

    def _set_model(self, model_id, model_scale):
        """Set the model id and scale. If not provided, choose one randomly."""
        reconfigure = False

        # Model ID
        if model_id is None:
            model_id = random_choice(self.model_ids, self._episode_rng)
        if model_id != self.model_id:
            reconfigure = True
        self.model_id = model_id
        self.model_info = self.model_db[self.model_id]

        # Scale
        if model_scale is None:
            model_scale = self.model_info.get("scale")
        if model_scale is None:
            bbox = self.model_info["bbox"]
            bbox_size = np.float32(bbox["max"]) - np.float32(bbox["min"])
            model_scale = 0.3 / max(bbox_size)  # hardcode
        if model_scale != self.model_scale:
            reconfigure = True
        self.model_scale = model_scale

        if "offset" in self.model_info:
            self.model_offset = np.float32(self.model_info["offset"])
        else:
            self.model_offset = -np.float32(bbox["min"]) * model_scale
        # Add a small clearance
        self.model_offset[2] += 0.01

        return reconfigure

    def _load_articulations(self):
        self.faucet: physx.PhysxArticulation = self._load_faucet()
        # Cache qpos to restore
        self._faucet_init_qpos = self.faucet.get_qpos()

        # Set friction and damping for all joints
        for joint in self.faucet.get_active_joints():
            joint.set_friction(1.0)
            joint.set_drive_property(0.0, 10.0)

        self._set_switch_links()

    def _load_faucet(self):
        loader = self._scene.create_urdf_loader()
        loader.scale = self.model_scale
        loader.fix_root_link = True

        model_dir = self.asset_root / str(self.model_id)
        urdf_path = model_dir / "mobility_cvx.urdf"

        density = self.model_info.get("density", 8e3)
        articulation: physx.PhysxArticulation = loader.load(str(urdf_path))
        loader.set_density(density)
        articulation.set_name("faucet")

        # TODO (stao): find out why we did this before
        set_articulation_render_material(
            articulation, color=hex2rgba("#AAAAAA"), metallic=1, roughness=0.4
        )

        return articulation

    def _set_switch_links(self):
        switch_link_names = []
        for semantic in self.model_info["semantics"]:
            if semantic[2] == "switch":
                switch_link_names.append(semantic[0])
        if len(switch_link_names) == 0:
            raise RuntimeError(self.model_id)
        self.switch_link_names = switch_link_names

        self.switch_links = []
        self.switch_links_mesh: List[trimesh.Trimesh] = []
        self.switch_joints = []
        all_links = self.faucet.get_links()
        all_joints = self.faucet.get_joints()

        for name in self.switch_link_names:
            link = get_obj_by_name(all_links, name)
            self.switch_links.append(link)

            # cache mesh
            link_mesh = get_component_mesh(link, False)
            self.switch_links_mesh.append(link_mesh)

            # hardcode
            joint = all_joints[link.get_index()]
            joint.set_friction(0.1)
            joint.set_drive_property(0.0, 2.0)
            self.switch_joints.append(joint)

    def _load_agent(self):
        super()._load_agent()

        links = self.agent.robot.get_links()
        self.lfinger = get_obj_by_name(links, "panda_leftfinger")
        self.rfinger = get_obj_by_name(links, "panda_rightfinger")
        self.lfinger_mesh = get_component_mesh(self.lfinger, False)
        self.rfinger_mesh = get_component_mesh(self.rfinger, False)

    def _initialize_articulations(self):
        p = np.zeros(3)
        p[:2] = self._episode_rng.uniform(-0.05, 0.05, [2])
        p[2] = self.model_offset[2]
        ori = self._episode_rng.uniform(-np.pi / 12, np.pi / 12)
        q = euler2quat(0, 0, ori)
        self.faucet.set_pose(Pose(p, q))
        # self.sink.set_pose(self.sink.pose * Pose(p=[p[0], p[1], 0.011], q=q))
        # self.wall.set_pose(self.wall.pose * Pose(p=[p[0], p[1], 0], q=q))

    def _initialize_task(self):
        self._set_target_link()
        self._set_init_and_target_angle()

        qpos = self._faucet_init_qpos.copy()
        qpos[self.target_joint_idx] = self.init_angle
        self.faucet.set_qpos(qpos)

        # -------------------------------------------------------------------------- #
        # For dense reward
        # -------------------------------------------------------------------------- #
        # NOTE(jigu): trimesh uses np.random to sample
        self.lfinger_pcd = trimesh.sample.sample_surface(
            self.lfinger_mesh, 256, seed=self._episode_seed
        )[0]
        self.rfinger_pcd = trimesh.sample.sample_surface(
            self.rfinger_mesh, 256, seed=self._episode_seed
        )[0]

        self.last_angle_diff = self.target_angle - self.current_angle

    def _set_target_link(self):
        n_switch_links = len(self.switch_link_names)
        idx = random_choice(np.arange(n_switch_links), self._episode_rng)

        self.target_link_name = self.switch_link_names[idx]
        self.target_link: physx.PhysxArticulationLinkComponent = self.switch_links[idx]
        self.target_joint: physx.PhysxArticulationJoint = self.switch_joints[idx]
        self.target_joint_idx = self.faucet.get_active_joints().index(self.target_joint)

        # x-axis is the revolute joint direction

        assert (
            self.target_joint.type == "revolute_unwrapped"
            or self.target_joint.type == "revolute"
        ), self.target_joint.type
        joint_pose = self.target_joint.get_global_pose().to_transformation_matrix()
        self.target_joint_axis = joint_pose[:3, 0]

        self.target_link_mesh: trimesh.Trimesh = self.switch_links_mesh[idx]

        # NOTE(jigu): trimesh uses np.random to sample
        self.target_link_pcd = trimesh.sample.sample_surface(
            self.target_link_mesh, 256, seed=self._episode_seed
        )[0]

        # NOTE(jigu): joint origin can be anywhere on the joint axis.
        # Thus, I use the center of mass at the beginning instead
        cmass_pose = self.target_link.pose * self.target_link.cmass_local_pose
        self.target_link_pos = cmass_pose.p

    def _set_init_and_target_angle(self):
        qmin, qmax = self.target_joint.get_limit()[0]
        if np.isinf(qmin):
            self.init_angle = 0
        else:
            self.init_angle = qmin
        if np.isinf(qmax):
            # maybe we can count statics
            self.target_angle = np.pi / 2
        else:
            self.target_angle = qmin + (qmax - qmin) * 0.9

        # The angle to go
        self.target_angle_diff = self.target_angle - self.init_angle

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(
            tcp_pose=vectorize_pose(self.agent.tcp.pose),
            target_angle_diff=np.array(self.target_angle_diff),
            target_joint_axis=self.target_joint_axis,
            target_link_pos=self.target_link_pos,
        )
        if self._obs_mode in ["state", "state_dict"]:
            angle_dist = self.target_angle - self.current_angle
            obs["angle_dist"] = np.array(angle_dist)
        return obs

    @property
    def current_angle(self):
        return self.faucet.get_qpos()[self.target_joint_idx]

    def evaluate(self, **kwargs):
        angle_dist = self.target_angle - self.current_angle
        return dict(success=angle_dist < 0, angle_dist=angle_dist)

    def _compute_distance(self):
        """Compute the distance between the tap and robot fingers."""
        T = self.target_link.pose.to_transformation_matrix()
        pcd = transform_points(T, self.target_link_pcd)
        T1 = self.lfinger.pose.to_transformation_matrix()
        T2 = self.rfinger.pose.to_transformation_matrix()
        pcd1 = transform_points(T1, self.lfinger_pcd)
        pcd2 = transform_points(T2, self.rfinger_pcd)
        # trimesh.PointCloud(np.vstack([pcd, pcd1, pcd2])).show()
        distance1 = cdist(pcd, pcd1)
        distance2 = cdist(pcd, pcd2)

        return min(distance1.min(), distance2.min())

    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            return 10.0

        distance = self._compute_distance()
        reward += 1 - np.tanh(distance * 5.0)

        angle_diff = self.target_angle - self.current_angle
        turn_reward_1 = 3 * (1 - np.tanh(max(angle_diff, 0) * 2.0))
        reward += turn_reward_1

        delta_angle = angle_diff - self.last_angle_diff
        if angle_diff > 0:
            turn_reward_2 = -np.tanh(delta_angle * 2)
        else:
            turn_reward_2 = np.tanh(delta_angle * 2)
        turn_reward_2 *= 5
        reward += turn_reward_2
        self.last_angle_diff = angle_diff

        return reward

    def compute_normalized_dense_reward(self, **kwargs):
        return self.compute_dense_reward(**kwargs) / 10.0

    def get_state_dict(self) -> np.ndarray:
        state = super().get_state_dict()
        return np.hstack([state, self.target_angle])

    def set_state_dict(self, state):
        self.target_angle = state[-1]
        super().set_state_dict(state[:-1])
        self.last_angle_diff = self.target_angle - self.current_angle
