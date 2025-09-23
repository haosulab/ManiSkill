from typing import Dict, List, Union

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, io_utils, sapien_utils
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig


@register_env(
    "TurnFaucet-v1",
    max_episode_steps=200,
    asset_download_ids=["partnet_mobility_faucet"],
)
class TurnFaucetEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["sparse", "none"]
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "fetch"]
    agent: Union[Panda, Fetch]
    TRAIN_JSON = PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_faucet_train.json"

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        reconfiguration_freq=None,
        num_envs=1,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.train_info = io_utils.load_json(self.TRAIN_JSON)
        self.all_model_ids = np.array(list(self.train_info.keys()))
        if reconfiguration_freq is None:
            # if not user set, we pick a number
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([-0.4, 0, 0.3], [0, 0, 0.1])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.5, 0.5, 1.0], [0.0, 0.0, 0.5])
        return CameraConfig("render_camera", pose=pose, width=512, height=512, fov=1)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()
        model_ids = self._batched_episode_rng.choice(self.all_model_ids)
        switch_link_ids = self._batched_episode_rng.randint(0, 2**31)

        self._faucets = []
        self._target_switch_links: List[Link] = []
        self.model_offsets = []
        for i, model_id in enumerate(model_ids):
            # partnet-mobility is a dataset source and the ids are the ones we sampled
            # we provide tools to easily create the articulation builder like so by querying
            # the dataset source and unique ID
            model_info = self.train_info[model_id]
            builder = articulations.get_articulation_builder(
                self.scene,
                f"partnet-mobility:{model_id}",
                urdf_config=dict(density=model_info.get("density", 8e3)),
            )
            builder.set_scene_idxs(scene_idxs=[i])
            builder.initial_pose = sapien.Pose(p=[0, 0, model_info["offset"][2]])
            faucet = builder.build(name=f"{model_id}-{i}")
            self.remove_from_state_dict_registry(faucet)
            for joint in faucet.active_joints:
                joint.set_friction(1.0)
                joint.set_drive_properties(0, 10.0)
            self.model_offsets.append(model_info["offset"])
            self._faucets.append(faucet)

            switch_link_names = []
            for j, semantic in enumerate(model_info["semantics"]):
                if semantic[2] == "switch":
                    switch_link_names.append(semantic[0])
            switch_link = faucet.links_map[
                switch_link_names[switch_link_ids[i] % len(switch_link_names)]
            ]
            self._target_switch_links.append(switch_link)
            switch_link.joint.set_friction(0.1)
            switch_link.joint.set_drive_properties(0.0, 2.0)
            sapien_utils.set_articulation_render_material(
                faucet._objs[0],
                color=sapien_utils.hex2rgba("#AAAAAA"),
                metallic=1,
                roughness=0.4,
            )

        self.faucet = Articulation.merge(self._faucets, name="faucet")
        self.add_to_state_dict_registry(self.faucet)
        self.target_switch_link = Link.merge(self._target_switch_links, name="switch")
        self.model_offsets = common.to_tensor(self.model_offsets, device=self.device)
        self.model_offsets[:, 2] += 0.01  # small clearance

        # self.handle_link_goal = actors.build_sphere(
        #     self.scene,
        #     radius=0.03,
        #     color=[0, 1, 0, 1],
        #     name="switch_link_goal",
        #     body_type="kinematic",
        #     add_collision=False,
        # )

        qlimits = self.target_switch_link.joint.get_limits()
        qmin, qmax = qlimits[:, 0], qlimits[:, 1]
        self.init_angle = qmin
        self.init_angle[torch.isinf(qmin)] = 0
        self.target_angle = qmin + (qmax - qmin) * 0.9
        self.target_angle[torch.isinf(qmax)] = torch.pi / 2
        # the angle to go
        self.target_angle_diff = self.target_angle - self.init_angle
        self.target_joint_axis = torch.zeros((self.num_envs, 3), device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.scene_builder.initialize(env_idx)
            b = len(env_idx)
            p = torch.zeros((b, 3))
            p[:, :2] = randomization.uniform(-0.05, 0.05, size=(b, 2))
            p[:, 2] = self.model_offsets[:, 2]
            # p[:, 2] = 0.5
            # ori = self._episode_rng.uniform(-np.pi / 12, np.pi / 12)
            q = randomization.random_quaternions(
                n=b, lock_x=True, lock_y=True, bounds=(-torch.pi / 12, torch.pi / 12)
            )
            self.faucet.set_pose(Pose.create_from_pq(p, q))

            # apply pose changes and update kinematics to get updated link poses.
            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene.px.step()
                self.scene._gpu_fetch_all()

            cmass_pose = (
                self.target_switch_link.pose * self.target_switch_link.cmass_local_pose
            )
            self.target_link_pos = cmass_pose.p
            joint_pose = (
                self.target_switch_link.joint.get_global_pose().to_transformation_matrix()
            )
            self.target_joint_axis[env_idx] = joint_pose[env_idx, :3, 0]
            # self.handle_link_goal.set_pose(cmass_pose)

    @property
    def current_angle(self):
        return self.target_switch_link.joint.qpos

    def evaluate(self):
        angle_dist = self.target_angle - self.current_angle
        return dict(success=angle_dist < 0, angle_dist=angle_dist)

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            target_angle_diff=self.target_angle_diff,
            target_joint_axis=self.target_joint_axis,
            target_link_pos=self.target_link_pos,
        )

        if "state" in self.obs_mode:
            angle_dist = self.target_angle - self.current_angle
            obs["angle_dist"] = angle_dist
        return obs

    # TODO (stao, tmu): finalize a dense reward that works for turn faucet
    # def compute_dense_reward(self, info, **kwargs):
    #     reward = 0.0

    #     if info["success"]:
    #         return 10.0

    #     distance = self._compute_distance()
    #     reward += 1 - np.tanh(distance * 5.0)

    #     # is_contacted = any(self.agent.check_contact_fingers(self.target_link))
    #     # if is_contacted:
    #     #     reward += 0.25

    #     angle_diff = self.target_angle - self.current_angle
    #     turn_reward_1 = 3 * (1 - np.tanh(max(angle_diff, 0) * 2.0))
    #     reward += turn_reward_1

    #     delta_angle = angle_diff - self.last_angle_diff
    #     if angle_diff > 0:
    #         turn_reward_2 = -np.tanh(delta_angle * 2)
    #     else:
    #         turn_reward_2 = np.tanh(delta_angle * 2)
    #     turn_reward_2 *= 5
    #     reward += turn_reward_2
    #     self.last_angle_diff = angle_diff

    #     return reward

    # def compute_normalized_dense_reward(
    #     self, obs: Any, action: torch.Tensor, info: Dict
    # ):
    #     max_reward = 10.0
    #     return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
