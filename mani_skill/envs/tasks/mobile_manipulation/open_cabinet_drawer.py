from typing import Any, Optional, Union

import numpy as np
import sapien
import sapien.physx as physx
import torch
import trimesh

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Fetch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.geometry.geometry import transform_points
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Articulation, Link, Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

CABINET_COLLISION_BIT = 29


# TODO (stao): we need to cut the meshes of all the cabinets in this dataset for gpu sim, there may be some wierd physics
# that may happen although it seems okay for state based RL
@register_env(
    "OpenCabinetDrawer-v1",
    asset_download_ids=["partnet_mobility_cabinet"],
    max_episode_steps=100,
)
class OpenCabinetDrawerEnv(BaseEnv):
    """
    **Task Description:**
    Use the Fetch mobile manipulation robot to move towards a target cabinet and open the target drawer out.

    **Randomizations:**
    - Robot is randomly initialized 1.6 to 1.8 meters away from the cabinet and positioned to face it
    - Robot's base orientation is randomized by -9 to 9 degrees
    - The cabinet selected to manipulate is randomly sampled from all PartnetMobility cabinets that have drawers
    - The drawer to open is randomly sampled from all drawers available to open

    **Success Conditions:**
    - The drawer is open at least 90% of the way, and the angular/linear velocities of the drawer link are small

    **Goal Specification:**
    - 3D goal position centered at the center of mass of the handle mesh on the drawer to open (also visualized in human renders with a sphere).
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/OpenCabinetDrawer-v1_rt.mp4"

    SUPPORTED_ROBOTS = ["fetch"]
    agent: Union[Fetch]
    handle_types = ["prismatic"]
    TRAIN_JSON = (
        PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_cabinet_drawer_train.json"
    )

    min_open_frac = 0.75

    def __init__(
        self,
        *args,
        robot_uids="fetch",
        robot_init_qpos_noise=0.02,
        reconfiguration_freq=None,
        num_envs=1,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        train_data = load_json(self.TRAIN_JSON)
        self.all_model_ids = np.array(list(train_data.keys()))
        # self.all_model_ids = np.array(["1004", "1004"])
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
        return SimConfig(
            spacing=5,
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**21, max_rigid_patch_count=2**19
            ),
        )

    @property
    def _default_sensor_configs(self):
        return []

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[-1.8, -1.3, 1.8], target=[-0.3, 0.5, 0])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[1, 0, 0]))

    def _load_scene(self, options: dict):
        self.ground = build_ground(self.scene)
        # temporarily turn off the logging as there will be big red warnings
        # about the cabinets having oblong meshes which we ignore for now.
        sapien.set_log_level("off")
        self._load_cabinets(self.handle_types)
        sapien.set_log_level("warn")
        from mani_skill.agents.robots.fetch import FETCH_WHEELS_COLLISION_BIT

        self.ground.set_collision_group_bit(
            group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1
        )
        self.ground.set_collision_group_bit(
            group=2, bit_idx=CABINET_COLLISION_BIT, bit=1
        )

    def _load_cabinets(self, joint_types: list[str]):
        # we sample random cabinet model_ids with numpy as numpy is always deterministic based on seed, regardless of
        # GPU/CPU simulation backends. This is useful for replaying demonstrations.
        model_ids = self._batched_episode_rng.choice(self.all_model_ids)
        link_ids = self._batched_episode_rng.randint(0, 2**31)

        self._cabinets: list[Articulation] = []
        handle_links: list[list[Link]] = []
        handle_links_meshes: list[list[trimesh.Trimesh]] = []
        for i, model_id in enumerate(model_ids):
            # partnet-mobility is a dataset source and the ids are the ones we sampled
            # we provide tools to easily create the articulation builder like so by querying
            # the dataset source and unique ID
            cabinet_builder = articulations.get_articulation_builder(
                self.scene, f"partnet-mobility:{model_id}"
            )
            cabinet_builder.set_scene_idxs(scene_idxs=[i])
            cabinet_builder.initial_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
            cabinet = cabinet_builder.build(name=f"{model_id}-{i}")
            self.remove_from_state_dict_registry(cabinet)
            # this disables self collisions by setting the group 2 bit at CABINET_COLLISION_BIT all the same
            # that bit is also used to disable collision with the ground plane
            for link in cabinet.links:
                link.set_collision_group_bit(
                    group=2, bit_idx=CABINET_COLLISION_BIT, bit=1
                )
            self._cabinets.append(cabinet)
            handle_links.append([])
            handle_links_meshes.append([])

            # TODO (stao): At the moment code for selecting semantic parts of articulations
            # is not very simple. Will be improved in the future as we add in features that
            # support part and mesh-wise annotations in a standard querable format
            for link, joint in zip(cabinet.links, cabinet.joints):
                if joint.type[0] in joint_types:
                    handle_links[-1].append(link)
                    # save the first mesh in the link object that correspond with a handle
                    handle_links_meshes[-1].append(
                        link.generate_mesh(
                            filter=lambda _, render_shape: "handle"
                            in render_shape.name,
                            mesh_name="handle",
                        )[0]
                    )

        # we can merge different articulations/links with different degrees of freedoms into a single view/object
        # allowing you to manage all of them under one object and retrieve data like qpos, pose, etc. all together
        # and with high performance. Note that some properties such as qpos and qlimits are now padded.
        self.cabinet = Articulation.merge(self._cabinets, name="cabinet")
        self.add_to_state_dict_registry(self.cabinet)
        self.handle_link = Link.merge(
            [links[link_ids[i] % len(links)] for i, links in enumerate(handle_links)],
            name="handle_link",
        )
        # store the position of the handle mesh itself relative to the link it is apart of
        self.handle_link_pos = common.to_tensor(
            np.array(
                [
                    meshes[link_ids[i] % len(meshes)].bounding_box.center_mass
                    for i, meshes in enumerate(handle_links_meshes)
                ]
            ),
            device=self.device,
        )

        self.handle_link_goal = actors.build_sphere(
            self.scene,
            radius=0.02,
            color=[0, 1, 0, 1],
            name="handle_link_goal",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
        )

    def _after_reconfigure(self, options):
        # To spawn cabinets in the right place, we need to change their z position such that
        # the bottom of the cabinet sits at z=0 (the floor). Luckily the partnet mobility dataset is made such that
        # the negative of the lower z-bound of the collision mesh bounding box is the right value

        # this code is in _after_reconfigure since retrieving collision meshes requires the GPU to be initialized
        # which occurs after the initial reconfigure call (after self._load_scene() is called)
        self.cabinet_zs = []
        for cabinet in self._cabinets:
            collision_mesh = cabinet.get_first_collision_mesh()
            self.cabinet_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.cabinet_zs = common.to_tensor(self.cabinet_zs, device=self.device)

        # get the qmin qmax values of the joint corresponding to the selected links
        target_qlimits = self.handle_link.joint.limits  # [b, 1, 2]
        qmin, qmax = target_qlimits[..., 0], target_qlimits[..., 1]
        self.target_qpos = qmin + (qmax - qmin) * self.min_open_frac

    def handle_link_positions(self, env_idx: Optional[torch.Tensor] = None):
        if env_idx is None:
            return transform_points(
                self.handle_link.pose.to_transformation_matrix().clone(),
                common.to_tensor(self.handle_link_pos, device=self.device),
            )
        return transform_points(
            self.handle_link.pose[env_idx].to_transformation_matrix().clone(),
            common.to_tensor(self.handle_link_pos[env_idx], device=self.device),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):

        with torch.device(self.device):
            b = len(env_idx)
            xy = torch.zeros((b, 3))
            xy[:, 2] = self.cabinet_zs[env_idx]
            self.cabinet.set_pose(Pose.create_from_pq(p=xy))

            # initialize robot
            if self.robot_uids == "fetch":
                qpos = torch.tensor(
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
                qpos = qpos.repeat(b).reshape(b, -1)
                dist = randomization.uniform(1.6, 1.8, size=(b,))
                theta = randomization.uniform(0.9 * torch.pi, 1.1 * torch.pi, size=(b,))
                xy = torch.zeros((b, 2))
                xy[:, 0] += torch.cos(theta) * dist
                xy[:, 1] += torch.sin(theta) * dist
                qpos[:, :2] = xy
                noise_ori = randomization.uniform(
                    -0.05 * torch.pi, 0.05 * torch.pi, size=(b,)
                )
                ori = (theta - torch.pi) + noise_ori
                qpos[:, 2] = ori
                self.agent.robot.set_qpos(qpos)
                self.agent.robot.set_pose(sapien.Pose())
            # close all the cabinets. We know beforehand that lower qlimit means "closed" for these assets.
            qlimits = self.cabinet.get_qlimits()  # [b, self.cabinet.max_dof, 2])
            self.cabinet.set_qpos(qlimits[env_idx, :, 0])
            self.cabinet.set_qvel(self.cabinet.qpos[env_idx] * 0)

            # NOTE (stao): This is a temporary work around for the issue where the cabinet drawers/doors might open
            # themselves on the first step. It's unclear why this happens on GPU sim only atm.
            # moreover despite setting qpos/qvel to 0, the cabinets might still move on their own a little bit.
            # this may be due to oblong meshes.
            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene.px.step()
                self.scene._gpu_fetch_all()

            self.handle_link_goal.set_pose(
                Pose.create_from_pq(p=self.handle_link_positions(env_idx))
            )

    def _after_control_step(self):
        # after each control step, we update the goal position of the handle link
        # for GPU sim we need to update the kinematics data to get latest pose information for up to date link poses
        # and fetch it, followed by an apply call to ensure the GPU sim is up to date
        if self.gpu_sim_enabled:
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
        self.handle_link_goal.set_pose(
            Pose.create_from_pq(p=self.handle_link_positions())
        )
        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()

    def evaluate(self):
        # even though self.handle_link is a different link across different articulations
        # we can still fetch a joint that represents the parent joint of all those links
        # and easily get the qpos value.
        open_enough = self.handle_link.joint.qpos >= self.target_qpos
        handle_link_pos = self.handle_link_positions()

        link_is_static = (
            torch.linalg.norm(self.handle_link.angular_velocity, axis=1) <= 1
        ) & (torch.linalg.norm(self.handle_link.linear_velocity, axis=1) <= 0.1)
        return {
            "success": open_enough & link_is_static,
            "handle_link_pos": handle_link_pos,
            "open_enough": open_enough,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )

        if "state" in self.obs_mode:
            obs.update(
                tcp_to_handle_pos=info["handle_link_pos"] - self.agent.tcp.pose.p,
                target_link_qpos=self.handle_link.joint.qpos,
                target_handle_pos=info["handle_link_pos"],
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_handle_dist = torch.linalg.norm(
            self.agent.tcp.pose.p - info["handle_link_pos"], axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_handle_dist)
        amount_to_open_left = torch.div(
            self.target_qpos - self.handle_link.joint.qpos, self.target_qpos
        )
        open_reward = 2 * (1 - amount_to_open_left)
        reaching_reward[
            amount_to_open_left < 0.999
        ] = 2  # if joint opens even a tiny bit, we don't need reach reward anymore
        # print(open_reward.shape)
        open_reward[info["open_enough"]] = 3  # give max reward here
        reward = reaching_reward + open_reward
        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        max_reward = 5.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward


@register_env("OpenCabinetDoor-v1", max_episode_steps=100)
class OpenCabinetDoorEnv(OpenCabinetDrawerEnv):
    TRAIN_JSON = (
        PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_cabinet_door_train.json"
    )
    handle_types = ["revolute", "revolute_unwrapped"]
