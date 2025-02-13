from pathlib import Path
from typing import Dict, Union

import numpy as np
import sapien.core as sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill import ASSET_DIR
from mani_skill.agents.robots import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, io_utils, sapien_utils
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@register_env(
    "AssemblingKits-v1", asset_download_ids=["assembling_kits"], max_episode_steps=200
)
class AssemblingKitsEnv(BaseEnv):
    """
    **Task Description:**
    The robot must pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.

    **Randomizations:**
    - the kit geometry is randomized, with different already inserted shapes and different holes affording insertion of specific shapes. (during reconfiguration)
    - the misplaced shape's geometry is sampled from one of 20 different shapes. (during reconfiguration)
    - the misplaced shape is randomly spawned anywhere on top of the board with a random z-axis rotation

    **Success Conditions:**
    - the misplaced shape is inserted completely into the correct slot
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/AssemblingKits-v1_rt.mp4"

    SUPPORTED_REWARD_MODES = ["sparse", "none"]
    SUPPORTED_ROBOTS = ["panda_wristcam"]
    agent: Union[PandaWristCam]

    def __init__(
        self,
        asset_root=f"{ASSET_DIR}/tasks/assembling_kits",
        robot_uids="panda_wristcam",
        num_envs=1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        self.asset_root = Path(asset_root)
        self._kit_dir = self.asset_root / "kits"
        self._models_dir = self.asset_root / "models"
        if not (self._kit_dir.exists() and self._models_dir.exists()):
            raise FileNotFoundError(
                "The objects/kits are not found."
                "Please download (ManiSkill) AssemblingKits assets:"
                "`python -m mani_skill.utils.download_asset assembling_kits`."
            )

        self._episode_json = io_utils.load_json(self.asset_root / "episodes.json")
        self._episodes = self._episode_json["episodes"]
        self.episode_idx = None

        self.symmetry = self._episode_json["config"]["symmetry"]
        self.color = self._episode_json["config"]["color"]
        self.object_scale = self._episode_json["config"]["object_scale"]
        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
        super().__init__(
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(max_rigid_contact_count=2**20)
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0.2, 0, 0.4], [0, 0, 0])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.3, 0.3, 0.8], [0.0, 0.0, 0.1])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        with torch.device(self.device):
            self.table_scene = TableSceneBuilder(self)
            self.table_scene.build()

            self.symmetry = common.to_tensor(self.symmetry)

            # sample some kits
            eps_idxs = self._batched_episode_rng.randint(0, len(self._episodes))
            pick_color_ids = self._batched_episode_rng.choice(len(self.color))
            other_color_ids = self._batched_episode_rng.choice(
                len(self.color), size=(10,)
            )

            kits = []
            objs_to_place = []
            all_other_objs = []
            self.object_ids = []
            self.goal_pos = np.zeros((self.num_envs, 3))
            self.goal_rot = np.zeros((self.num_envs,))

            for i, eps_idx in enumerate(eps_idxs):
                scene_idxs = [i]
                episode = self._episodes[eps_idx]

                # get the kit builder and the goal positions/rotations of all other objects
                (
                    kit_builder,
                    object_goal_pos,
                    object_goal_rot,
                ) = self._get_kit_builder_and_goals(episode["kit"])
                kit = (
                    kit_builder.set_scene_idxs(scene_idxs)
                    .set_initial_pose(sapien.Pose([0, 0, 0.01]))
                    .build_static(f"kit_{i}")
                )
                kits.append(kit)
                # create the object to place and make it dynamic
                obj_to_place = (
                    self._get_object_builder(
                        episode["obj_to_place"], color_id=pick_color_ids[i]
                    )
                    .set_scene_idxs(scene_idxs)
                    .set_initial_pose(sapien.Pose(p=[0, 0, 0.1]))
                    .build(f"obj_{i}")
                )
                self.object_ids.append(episode["obj_to_place"])
                objs_to_place.append(obj_to_place)

                # create all other objects and leave them as static as they do not need to be manipulated
                other_objs = [
                    self._get_object_builder(
                        obj_id, static=True, color_id=other_color_ids[i, j]
                    )
                    .set_scene_idxs(scene_idxs)
                    .set_initial_pose(
                        sapien.Pose(
                            object_goal_pos[obj_id],
                            q=euler2quat(0, 0, object_goal_rot[obj_id]),
                        )
                    )
                    .build_static(f"in_place_obj_{i}_{j}")
                    for j, obj_id in enumerate(episode["obj_in_place"])
                ]
                all_other_objs.append(other_objs)

                # save the goal position and z-axis rotation of the object to place
                self.goal_pos[i] = object_goal_pos[episode["obj_to_place"]]
                self.goal_rot[i] = object_goal_rot[episode["obj_to_place"]]
            self.obj = Actor.merge(objs_to_place)
            self.object_ids = torch.tensor(self.object_ids, dtype=int)
            self.goal_pos = common.to_tensor(self.goal_pos)
            self.goal_rot = common.to_tensor(self.goal_rot)

    def _parse_json(self, path):
        """Parse kit JSON information and return the goal positions and rotations"""
        kit_json = io_utils.load_json(path)
        # the final 3D goal position of the objects
        object_goal_pos = {
            o["object_id"]: common.to_numpy(o["pos"]) for o in kit_json["objects"]
        }
        # the final goal z-axis rotation of the objects
        objects_goal_rot = {o["object_id"]: o["rot"] for o in kit_json["objects"]}
        return object_goal_pos, objects_goal_rot

    def _get_kit_builder_and_goals(self, kit_id: str):
        object_goal_pos, objects_goal_rot = self._parse_json(
            self._kit_dir / f"{kit_id}.json"
        )
        builder = self.scene.create_actor_builder()
        kit_path = str(self._kit_dir / f"{kit_id}.obj")
        builder.add_nonconvex_collision_from_file(kit_path)
        builder.add_visual_from_file(
            kit_path,
            material=sapien.render.RenderMaterial(
                base_color=[0.27807487, 0.20855615, 0.16934046, 1.0],
                roughness=0.5,
                specular=0.0,
            ),
        )
        return builder, object_goal_pos, objects_goal_rot

    def _get_object_builder(
        self, object_id: str, static: bool = False, color_id: int = 0
    ):
        collision_path = self._models_dir / "collision" / f"{object_id:02d}.obj"
        visual_path = self._models_dir / "visual" / f"{object_id:02d}.obj"

        builder = self.scene.create_actor_builder()
        if static:
            builder.add_nonconvex_collision_from_file(
                str(collision_path), scale=self.object_scale
            )
        else:
            builder.add_multiple_convex_collisions_from_file(
                str(collision_path), scale=self.object_scale
            )
        builder.add_visual_from_file(
            str(visual_path),
            scale=self.object_scale,
            material=sapien.render.RenderMaterial(
                base_color=self.color[color_id],
                roughness=0.1,
                specular=0.0,
            ),
        )
        return builder

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, 0] = torch.rand((b,)) * 0.2 - 0.1
            xyz[:, 1] = torch.rand((b,)) * 0.364 - 0.364 / 2
            xyz[:, 2] = 0.02
            q = randomization.random_quaternions(
                b, device=self.device, lock_x=True, lock_y=True
            )
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=q))

    def _check_pos_diff(self, pos_eps=2e-2):
        pos_diff = self.goal_pos[:, :2] - self.obj.pose.p[:, :2]
        pos_diff_norm = torch.linalg.norm(pos_diff, axis=1)
        return pos_diff, pos_diff_norm, pos_diff_norm < pos_eps

    def _check_rot_diff(self, rot_eps=np.deg2rad(4)):
        rot = rotation_conversions.matrix_to_euler_angles(
            rotation_conversions.quaternion_to_matrix(self.obj.pose.q), "XYZ"
        )[:, -1]
        rot_diff = torch.zeros((self.num_envs), dtype=torch.float, device=self.device)

        has_symmetries = self.symmetry[self.object_ids] > 0
        rot_diff_sym = torch.abs(rot - self.goal_rot) % self.symmetry[self.object_ids]
        has_half_symmetries = rot_diff_sym > self.symmetry[self.object_ids] / 2

        rot_diff[has_symmetries] = rot_diff_sym[has_symmetries]
        rot_diff[has_half_symmetries] = (
            self.symmetry[self.object_ids][has_half_symmetries]
            - rot_diff_sym[has_half_symmetries]
        )
        return rot_diff, rot_diff < rot_eps

    def _check_in_slot(self, obj: Actor, height_eps=3e-3):
        return obj.pose.p[:, 2] < height_eps

    def evaluate(self) -> dict:
        pos_diff, pos_diff_norm, pos_correct = self._check_pos_diff()
        rot_diff, rot_correct = self._check_rot_diff()
        in_slot = self._check_in_slot(self.obj)
        return {
            "pos_diff": pos_diff,
            "pos_diff_norm": pos_diff_norm,
            "pos_correct": pos_correct,
            "rot_diff": rot_diff,
            "rot_correct": rot_correct,
            "in_slot": in_slot,
            "success": pos_correct & rot_correct & in_slot,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                obj_pose=self.obj.pose.raw_pose,
                tcp_to_obj_pos=self.obj.pose.p - self.agent.tcp.pose.p,
                goal_pos=self.goal_pos,
                goal_rot=self.goal_rot,
                obj_to_goal_pos=self.goal_pos - self.obj.pose.p,
            )
        return obs
