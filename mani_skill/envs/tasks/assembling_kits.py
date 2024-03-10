from pathlib import Path
from typing import Union

import numpy as np
import sapien.core as sapien
import torch
from transforms3d.euler import euler2quat, quat2euler

from mani_skill import ASSET_DIR, format_path
from mani_skill.agents.robots import PandaRealSensed435
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.scene_builder.table.table_scene_builder import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose


@register_env("AssemblingKits-v1", max_episode_steps=200)
class AssemblingKitsEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["sparse", "none"]
    SUPPORTED_ROBOTS = ["panda_realsensed435"]
    agent: Union[PandaRealSensed435]

    def __init__(
        self,
        asset_root=f"{ASSET_DIR}/tasks/assembling_kits",
        robot_uids="panda_realsensed435",
        num_envs=1,
        **kwargs,
    ):
        self.asset_root = Path(asset_root)
        self._kit_dir = self.asset_root / "kits"
        self._models_dir = self.asset_root / "models"
        if not (self._kit_dir.exists() and self._models_dir.exists()):
            raise FileNotFoundError(
                "The objects/kits are not found."
                "Please download (ManiSkill2) AssemblingKits assets:"
                "`python -m mani_skill.utils.download_asset assembling_kits`."
            )

        self._episode_json = load_json(self.asset_root / "episodes.json")
        self._episodes = self._episode_json["episodes"]
        self.episode_idx = None

        self.symmetry = self._episode_json["config"]["symmetry"]
        self.color = self._episode_json["config"]["color"]
        self.object_scale = self._episode_json["config"]["object_scale"]
        reconfiguration_freq = 0
        if num_envs == 1:
            reconfiguration_freq = 1
        super().__init__(
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

    # def reset(self, seed=None, options=None):
    #     if options is None:
    #         options = dict()
    #     self.set_episode_rng(seed)
    #     episode_idx = options.pop("episode_idx", None)
    #     reconfigure = options.pop("reconfigure", False)
    #     if episode_idx is None:
    #         episode_idx = self._episode_rng.randint(len(self._episodes))
    #     if self.episode_idx != episode_idx:
    #         reconfigure = True
    #     self.episode_idx = episode_idx
    #     options["reconfigure"] = reconfigure

    #     episode = self._episodes[episode_idx]
    #     self.kit_id: int = episode["kit"]
    #     self.spawn_pos = np.float32(episode["spawn_pos"])
    #     self.object_id: int = episode["obj_to_place"]
    #     self._other_objects_id: List[int] = episode["obj_in_place"]

    #     return super().reset(seed=self._episode_seed, options=options)

    def _load_scene(self):
        with torch.device(self.device):
            self.table_scene = TableSceneBuilder(self)
            self.table_scene.build()

            # sample some kits
            eps_idxs = np.arange(0, len(self._episodes))
            self._episode_rng.shuffle(eps_idxs)
            eps_idxs = eps_idxs[: self.num_envs]
            print(eps_idxs)
            kits = []
            objs_to_place = []
            all_other_objs = []
            self.goal_pos = torch.zeros((self.num_envs, 3))
            self.goal_rot = torch.zeros((self.num_envs,))
            for i, eps_idx in enumerate(eps_idxs):
                scene_mask = np.zeros((self.num_envs,), dtype=bool)
                scene_mask[i] = True
                episode = self._episodes[eps_idx]

                # get the kit builder and the goal positions/rotations of all other objects
                (
                    kit_builder,
                    object_goal_pos,
                    object_goal_rot,
                ) = self._get_kit_builder_and_goals(episode["kit"])
                kit = (
                    kit_builder.set_scene_mask(scene_mask)
                    .set_initial_pose(sapien.Pose([0, 0, 0.01]))
                    .build_static(f"kit_{i}")
                )
                kits.append(kit)

                # create the object to place and make it dynamic
                obj_to_place = (
                    self._get_object_builder(episode["obj_to_place"])
                    .set_scene_mask(scene_mask)
                    .build(f"obj_{i}")
                )
                objs_to_place.append(obj_to_place)

                # create all othre objects and leave them as static as they do not need to be manipulated
                other_objs = [
                    self._get_object_builder(obj_id, static=True)
                    .set_scene_mask(scene_mask)
                    .set_initial_pose(
                        sapien.Pose(
                            object_goal_pos[obj_id],
                            q=euler2quat(0, 0, object_goal_rot[obj_id]),
                        )
                    )
                    .build_static(f"in_place_obj_{i}")
                    for i, obj_id in enumerate(episode["obj_in_place"])
                ]
                all_other_objs.append(other_objs)

                # save the goal position and z-axis rotation of the object to place
                self.goal_pos[i] = object_goal_pos[i]
                self.goal_rot[i] = object_goal_rot[i]
            self.obj = Actor.merge(objs_to_place)

    def _parse_json(self, path):
        """Parse kit JSON information and return the goal positions and rotations"""
        kit_json = load_json(path)
        # the final 3D goal position of the objects
        object_goal_pos = {
            o["object_id"]: torch.tensor(
                o["pos"], dtype=torch.float, device=self.device
            )
            for o in kit_json["objects"]
        }
        # the final goal z-axis rotation of the objects
        objects_goal_rot = {o["object_id"]: o["rot"] for o in kit_json["objects"]}
        return object_goal_pos, objects_goal_rot

    def _get_kit_builder_and_goals(self, kit_id: str):
        object_goal_pos, objects_goal_rot = self._parse_json(
            self._kit_dir / f"{kit_id}.json"
        )

        builder = self._scene.create_actor_builder()
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

    def _get_object_builder(self, object_id: str, static: bool = False):
        collision_path = self._models_dir / "collision" / f"{object_id:02d}.obj"
        visual_path = self._models_dir / "visual" / f"{object_id:02d}.obj"

        builder = self._scene.create_actor_builder()
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
                base_color=self.color[self._episode_rng.choice(len(self.color))],
                roughness=0.1,
                specular=0.0,
            ),
        )
        return builder

    def _initialize_episode(self, env_idx: torch.Tensor):
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

    def _initialize_task(self):
        self._obj_init_pos = np.float32(self.spawn_pos)
        self._obj_goal_pos = np.float32(self.objects_pos[self.object_id])

    def _get_obs_extra(self):
        obs = dict(
            tcp_pose=vectorize_pose(self.tcp.pose),
            obj_init_pos=self._obj_init_pos,
            obj_goal_pos=self._obj_goal_pos,
        )
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                obj_pose=vectorize_pose(self.obj.pose),
                tcp_to_obj_pos=self.obj.pose.p - self.tcp.pose.p,
                obj_to_goal_pos=self.objects_pos[self.object_id] - self.obj.pose.p,
            )
        return obs

    def _check_pos_diff(self, pos_eps=2e-2):
        pos_diff = self.objects_pos[self.object_id][:2] - self.obj.get_pose().p[:2]
        pos_diff_norm = np.linalg.norm(pos_diff)
        return pos_diff, pos_diff_norm, pos_diff_norm < pos_eps

    def _check_rot_diff(self, rot_eps=np.deg2rad(4)):
        rot = quat2euler(self.obj.get_pose().q)[-1]  # Check z-axis rotation
        rot_diff = 0
        if self.symmetry[self.object_id] > 0:
            rot_diff = (
                np.abs(rot - self.objects_rot[self.object_id])
                % self.symmetry[self.object_id]
            )
            if rot_diff > (self.symmetry[self.object_id] / 2):
                rot_diff = self.symmetry[self.object_id] - rot_diff
        return rot_diff, rot_diff < rot_eps

    # def _check_in_slot(self, obj: sapien.Actor, height_eps=3e-3):
    #     return obj.pose.p[2] < height_eps

    def evaluate(self, **kwargs) -> dict:
        return {}
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
            "success": pos_correct and rot_correct and in_slot,
        }

    @property
    def _sensor_configs(self):
        pose = sapien_utils.look_at([0.2, 0, 0.4], [0, 0, 0])
        return [
            CameraConfig("base_camera", pose.p, pose.q, 128, 128, np.pi / 2, 0.01, 100)
        ]

    @property
    def _human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.3, 0.3, 0.8], [0.0, 0.0, 0.1])
        return CameraConfig("render_camera", pose.p, pose.q, 512, 512, 1, 0.01, 100)
