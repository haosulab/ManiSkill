from collections import OrderedDict
from typing import Any, Dict, List

import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.actors import build_sphere
from mani_skill.utils.building.articulations import (
    MODEL_DBS,
    _load_partnet_mobility_dataset,
    build_preprocessed_partnet_mobility_articulation,
)
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.geometry.geometry import transform_points
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.link import Link


# TODO (stao): we need to cut the meshes of all the cabinets in this dataset for gpu sim, not registering task for now
# @register_env("OpenCabinetDrawer-v1", max_episode_steps=100)
class OpenCabinetDrawerEnv(BaseEnv):

    handle_types = ["prismatic"]

    def __init__(
        self,
        *args,
        robot_uids="fetch",
        robot_init_qpos_noise=0.02,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        _load_partnet_mobility_dataset()
        self.all_model_ids = np.array(
            list(MODEL_DBS["PartnetMobility"]["model_data"].keys())
        )
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _sensor_configs(self):
        pose = sapien_utils.look_at(eye=[-2.5, -1.5, 1.8], target=[-0.3, 0.5, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[-2.3, -1.5, 1.8], target=[-0.3, 0.5, 0])
        # TODO (stao): how much does far affect rendering speed?
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self):
        self.ground = build_ground(self._scene)
        self._load_cabinets(self.handle_types)

        from mani_skill.agents.robots.fetch import FETCH_UNIQUE_COLLISION_BIT

        # TODO (stao) (arth): is there a better way to model robots in sim. This feels very unintuitive.
        for obj in self.ground._objs:
            cs = obj.find_component_by_type(
                sapien.physx.PhysxRigidStaticComponent
            ).get_collision_shapes()[0]
            cg = cs.get_collision_groups()
            cg[2] |= FETCH_UNIQUE_COLLISION_BIT
            cg[2] |= 1 << 29  # make ground ignore collisions with the cabinets
            cs.set_collision_groups(cg)

    def _load_cabinets(self, joint_types: List[str]):
        rand_idx = torch.randperm(len(self.all_model_ids))
        model_ids = self.all_model_ids[rand_idx]
        model_ids = np.concatenate(
            [model_ids] * np.ceil(self.num_envs / len(self.all_model_ids)).astype(int)
        )[: self.num_envs]
        cabinets = []
        self.cabinet_heights = []
        handle_links: List[List[Link]] = []
        handle_links_meshes: List[List[Any]] = []
        for i, model_id in enumerate(model_ids):
            scene_mask = np.zeros(self.num_envs, dtype=bool)
            scene_mask[i] = True
            cabinet, metadata = build_preprocessed_partnet_mobility_articulation(
                self._scene, model_id, name=f"{model_id}-{i}", scene_idxs=scene_mask
            )
            # TODO (stao): since we processed the assets we know that the bounds[0, 1] is the actual height to set the object at
            # but in future we will store a visual origin offset so we can place them by using the actual bbox height / 2
            # TODO (stao): ask fanbo, it seems loading links does not load the pose correclty on the cpu object?
            self.cabinet_heights.append((-metadata.bbox.bounds[0, 1]))
            handle_links.append([])
            handle_links_meshes.append([])
            # NOTE (stao): interesting future project similar to some kind of quality diversity is accelerating policy learning by dynamically shifting distribution of handles/cabinets being trained on.
            for link, joint in zip(cabinet.links, cabinet.joints):
                if joint.type[0] in joint_types:
                    handle_links[-1].append(link)
                    handle_links_meshes[-1].append(
                        link.generate_mesh(lambda _, x: "handle" in x.name, "handle")[0]
                    )
            cabinets.append(cabinet)

        # we can merge different articulations with different degrees of freedoms as done below
        # allowing you to manage all of them under one object and retrieve data like qpos, pose, etc. all together
        # and with high performance. Note that some properties such as qpos and qlimits are now padded.
        self.cabinet = Articulation.merge(cabinets, name="cabinet")

        self.cabinet_metadata = metadata
        # list of list of links and meshes. handle_links[i][j] is the jth handle of the ith cabinet
        self.handle_links = handle_links
        self.handle_links_meshes = handle_links_meshes

        self.handle_link_goal = build_sphere(
            self._scene,
            radius=0.05,
            color=[0, 1, 0, 1],
            name="handle_link_goal",
            body_type="kinematic",
            add_collision=False,
        )
        self._hidden_objects.append(self.handle_link_goal)

    def _initialize_episode(self, env_idx: torch.Tensor):
        # TODO (stao): Clean up this code and try to batch / cache more if possible.
        # And support partial resets
        with torch.device(self.device):
            b = len(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, 2] = torch.tensor(self.cabinet_heights)
            self.cabinet.set_pose(Pose.create_from_pq(p=xyz))

            # this is not pure uniform but for faster initialization to deal with different cabinet DOFs we just sample 0 to 10000 and take the modulo which is close enough
            self.link_indices = torch.randint(
                0, 10000, size=(len(self.handle_links),)
            ) % torch.tensor([len(x) for x in self.handle_links], dtype=int)

            self.handle_link = Link.merge(
                [x[self.link_indices[i]] for i, x in enumerate(self.handle_links)],
                self.cabinet,
            )
            # cache/save the slice to reference the qpos and qvel of the link/joint we want to open
            index_q = []
            for art, link in zip(self.cabinet._objs, self.handle_link._objs):
                index_q.append(art.active_joints.index(link.joint))
            index_q = torch.tensor(index_q, dtype=int)
            self.target_qpos_idx = (torch.arange(0, b), index_q)
            # TODO (stao): For performance improvement, one can save relative position of link handles ahead of time.
            handle_link_positions = sapien_utils.to_tensor(
                np.array(
                    [
                        x[self.link_indices[i]].bounding_box.center_mass
                        for i, x in enumerate(self.handle_links_meshes)
                    ]
                )
            ).float()  # (N, 3)

            # the three lines here are necessary to update all link poses whenever qpos and root pose of articulation change
            # that way you can use the correct link poses as done below for your task.
            self._scene._gpu_apply_all()
            self._scene.px.gpu_update_articulation_kinematics()
            self._scene._gpu_fetch_all()

            handle_link_positions = transform_points(
                self.handle_link.pose.to_transformation_matrix().clone(),
                handle_link_positions,
            )
            self.handle_link_goal.set_pose(Pose.create_from_pq(p=handle_link_positions))
            # close all the cabinets. We know beforehand that lower qlimit means "closed" for these assets.
            qlimits = self.cabinet.get_qlimits()  # [N, self.cabinet.max_dof, 2])
            self.cabinet.set_qpos(qlimits[:, :, 0])

            # get the qmin qmax values of the joint corresponding to the selected links
            target_qlimits = qlimits[self.target_qpos_idx]
            qmin, qmax = target_qlimits[:, 0], target_qlimits[:, 1]
            self.target_qpos = qmin + (qmax - qmin) * 0.9

            # initialize robot
            if self.robot_uids == "panda":
                self.agent.robot.set_qpos(self.agent.robot.qpos * 0)
                self.agent.robot.set_pose(Pose.create_from_pq(p=[-1, 0, 0]))
            elif self.robot_uids == "fetch":
                qpos = np.array(
                    [
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        -np.pi / 4,
                        0,
                        np.pi / 4,
                        0,
                        np.pi / 3,
                        0,
                        0.015,
                        0.015,
                    ]
                )
                self.agent.reset(qpos)
                self.agent.robot.set_pose(sapien.Pose([-1.5, 0, 0]))

            # NOTE (stao): This is a temporary work around for the issue where the cabinet drawers/doors might open themselves on the first step. It's unclear why this happens on GPU sim only atm.
            self._scene._gpu_apply_all()
            self._scene.px.step()
            self.cabinet.set_qpos(qlimits[:, :, 0])

    ### Useful properties ###

    def evaluate(self):
        link_qpos = self.cabinet.qpos[self.target_qpos_idx]
        self.cabinet.qvel[self.target_qpos_idx]
        open_enough = link_qpos >= self.target_qpos
        return {"success": open_enough, "link_qpos": link_qpos}

    def _get_obs_extra(self, info: Dict):
        # TODO (stao): fix the observation to be correct when in state or not mode
        # moreover also check if hiding goal visual affects the observation data as well
        obs = OrderedDict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            target_handle_pos=self.handle_link_goal.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                tcp_to_handle_pos=self.handle_link_goal.pose.p - self.agent.tcp.pose.p,
                target_link_qpos=self.cabinet.qpos[self.target_qpos_idx],
                # obs_pose=self.cube.pose.raw_pose,
                # tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp.pose.p,
                # obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        tcp_to_handle_dist = torch.linalg.norm(
            self.agent.tcp.pose.p - self.handle_link.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_handle_dist)
        reward = reaching_reward
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward


# @register_env("OpenCabinetDoor-v1", max_episode_steps=200)
class OpenCabinetDoorEnv(OpenCabinetDrawerEnv):
    handle_types = ["revolute"]
