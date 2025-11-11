from typing import Any, Dict, List, Optional, Union

import numpy as np
import sapien
import sapien.physx as physx
import torch
import trimesh

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.geometry.geometry import transform_points
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Articulation, Link, Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.envs.tasks.tabletop.colosseum_v2_versions.colosseum_v2_env_utils import get_human_render_camera_config, REALSENSE_DEPTH_FOV_VERTICAL_RAD, SHADER
from mani_skill.envs.distraction_set import DistractionSet

CABINET_COLLISION_BIT = 29


# TODO (stao): we need to cut the meshes of all the cabinets in this dataset for gpu sim, there may be some wierd physics
# that may happen although it seems okay for state based RL
@register_env(
    "OpenDrawer-v1",
    asset_download_ids=["partnet_mobility_cabinet"],
    max_episode_steps=100,
)
class OpenDrawerV1Env(BaseEnv):
    """
    **Task Description:**
    Use the Panda open the target drawer out.

    Largely borrowed from mani_skill/envs/tasks/mobile_manipulation/open_cabinet_drawer.py


    Appropriate pointcloud bounds:
    x: [-0.4, 0.1] 
    y: [-0.3, 0.3] 
    z: [0.4, 0.8]

    
    Only used for calculating the pointcloud bounds:
        center: (-0.2, 0, 0.6)
        offsets wrt. center:
            x: [-0.2, 0.3]
            y: [-0.3, 0.3]
            z: [-.2, 0.2]
    """

    SUPPORTED_ROBOTS = ["panda"]
    agent: Union[Panda]
    handle_types = ["prismatic"]
    TRAIN_JSON = (
        PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_cabinet_drawer_train.json"
    )
    CABINET_X_LIMS = [0.15, 0.25]
    # ^ Starts getting planning failures above 0.25
    CABINET_Y_LIMS = [-0.05, 0.05]

    min_open_frac = 0.5

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        **kwargs,
    ):
        assert "camera_width" in kwargs and "camera_height" in kwargs, "camera_width and camera_height must be provided"
        assert "distraction_set" in kwargs, "distraction_set must be provided"
        self._human_render_shader = kwargs.pop("human_render_shader", None)
        self._camera_width = kwargs.pop("camera_width")
        self._camera_height = kwargs.pop("camera_height")
        self._distraction_set = kwargs.pop("distraction_set")
        if isinstance(self._distraction_set, dict):
            self._distraction_set = DistractionSet(**self._distraction_set)
        # 
        self.robot_init_qpos_noise = robot_init_qpos_noise
        # TRAIN_JSON.keys(): ['1000' '1004' '1005' '1013' '1016' '1021' '1024' '1027' '1032' '1033'
        #  '1035' '1038' '1040' '1044' '1045' '1052' '1054' '1056' '1061' '1063'
        #  '1066' '1067' '1076' '1079' '1082']
        # Good, but lower drawer: 1004
        # Missing top: 1038, 1040, 1045,  1052, 1054 
        # Drawer is too small: 1005, 1000, 1013, 1016,1021, 1024, 1027, 1032, 1033, 1035
        # self._model_id = 1005 # don't like the color
        self._model_id = 1027

        super().__init__(
            *args,
            robot_uids=robot_uids,
            **kwargs,
        )

    @property
    def _default_human_render_camera_configs(self):
        return get_human_render_camera_config(eye=[-0.2, 0.5, 1.1], target=[-0.1, 0, 0.5], shader=self._human_render_shader)

    @property
    def _default_sensor_configs(self):
        target = [-0.2, 0, 0.5]
        pose_center = sapien_utils.look_at(eye=[-0.5, 0.0, 1.25], target=target)
        pose_left = sapien_utils.look_at(eye=[-0.2, 0.5, 1.1], target=target)
        pose_right = sapien_utils.look_at(eye=[-0.2, -0.5, 1.1], target=target)
        cfgs = [
            CameraConfig(
                uid="camera_center",
                pose=pose_center,
                width=self._camera_width,
                height=self._camera_height,
                fov=REALSENSE_DEPTH_FOV_VERTICAL_RAD,
                near=0.01,
                far=100,
                shader_pack=SHADER,
            ),
            CameraConfig(
                uid="camera_left",
                pose=pose_left,
                width=self._camera_width,
                height=self._camera_height,
                fov=REALSENSE_DEPTH_FOV_VERTICAL_RAD,
                near=0.01,
                far=100,
                shader_pack=SHADER,
            ),
            CameraConfig(
                uid="camera_right",
                pose=pose_right,
                width=self._camera_width,
                height=self._camera_height,
                fov=REALSENSE_DEPTH_FOV_VERTICAL_RAD,
                near=0.01,
                far=100,
                shader_pack=SHADER,
            )]
        return self._distraction_set.update_camera_configs(cfgs)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))


    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        # temporarily turn off the logging as there will be big red warnings
        # about the cabinets having oblong meshes which we ignore for now.
        sapien.set_log_level("off")
        self._load_cabinets(self.handle_types)
        sapien.set_log_level("warn")
        self._hidden_objects.append(self.handle_link_goal)


    def _load_cabinets(self, joint_types: List[str]):
        # we sample random cabinet model_ids with numpy as numpy is always deterministic based on seed, regardless of
        # GPU/CPU simulation backends. This is useful for replaying demonstrations.
        # link_ids = self._batched_episode_rng.randint(0, 2**31)
        link_ids = [0]
        # ^ fix this so that we use the same drawer every time

        self._cabinets = []
        handle_links: List[List[Link]] = []
        handle_links_meshes: List[List[trimesh.Trimesh]] = []

        # partnet-mobility is a dataset source and the ids are the ones we sampled
        # we provide tools to easily create the articulation builder like so by querying
        # the dataset source and unique ID
        cabinet_builder = articulations.get_articulation_builder(
            self.scene, f"partnet-mobility:{self._model_id}"
        )
        # cabinet_builder.set_scene_idxs(scene_idxs=[0])
        cabinet_builder.initial_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
        cabinet = cabinet_builder.build(name=f"cabinet-{self._model_id}")
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
            cabinet_x_range = self.CABINET_X_LIMS[1] - self.CABINET_X_LIMS[0]
            cabinet_y_range = self.CABINET_Y_LIMS[1] - self.CABINET_Y_LIMS[0]
            xy[:, 0] = torch.rand(b) * cabinet_x_range + self.CABINET_X_LIMS[0]
            xy[:, 1] = torch.rand(b) * cabinet_y_range + self.CABINET_Y_LIMS[0]

            xy[:, 2] = self.cabinet_zs[env_idx]
            self.cabinet.set_pose(Pose.create_from_pq(p=xy))


            # initialize robot
            qpos_0 = np.array([-0.13595445, -1.2611351, 0.24094589, -2.9000182, 2.5728698, 3.0259767, 0.029944034, 0.039999813, 0.03999985]) # final two are gripper (start open)
            # ^ Copied from visualizer
            self.table_scene.initialize(env_idx, table_z_rotation_angle=np.pi, qpos_0=qpos_0)
            # ^ table_z_rotation_angle=np.pi rotates the table 90 degrees from default so that the cabinet has more table space behind it


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