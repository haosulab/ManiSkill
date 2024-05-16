import os
from pathlib import Path
from typing import List, Optional

import cv2
import gymnasium as gym
import numpy as np
import sapien.physx as physx
import torch

from mani_skill import ASSET_DIR
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.types import SimConfig


class BaseDigitalTwinEnv(BaseEnv):
    """Base Environment class for easily setting up evaluation digital twins for real2sim and sim2real

    This is based on the [SIMPLER](https://simpler-env.github.io/) and has the following tricks for
    making accurate simulated environments of real world datasets

    Greenscreening: TODO

    Texture Matching: TODO

    Note that this is not a general purpose system for building digital twins you can train and then transfer
    to the real world. This is designed to support fast evaluation in simulation of real world policies.

    """

    SUPPORTED_OBS_MODES = ["none", "state", "state_dict", "rgb", "rgbd"]

    rgb_overlay_path: Optional[str] = None
    """path to the file to place on the greenscreen"""
    rgb_always_overlay_objects: List[str] = []
    """List of names of actors/links that should not be covered by the greenscreen"""
    rgb_overlay_cameras: List[str]
    """Cameras to do greenscreening over when fetching image observations"""
    rgb_overlay_mode: str = (
        "background"  # 'background' or 'object' or 'debug' or combinations of them
    )
    """which RGB overlay mode to use during the greenscreen process"""

    def __init__(self, **kwargs):
        # Load the "greenscreen" image, which is used to overlay the background portions of simulation observation
        if self.rgb_overlay_path is not None:
            if not os.path.exists(self.rgb_overlay_path):
                raise FileNotFoundError(
                    f"rgb_overlay_path {self.rgb_overlay_path} is not found."
                    "If you installed this repo through 'pip install .' , "
                    "you can download this directory https://github.com/simpler-env/ManiSkill2_real2sim/tree/main/data to get the real-world image overlay assets. "
                )
            self.rgb_overlay_img = cv2.cvtColor(
                cv2.imread(self.rgb_overlay_path), cv2.COLOR_BGR2RGB
            )  # (H, W, 3); float32
        else:
            self.rgb_overlay_img = None

        super().__init__(**kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_scene(self, options: dict):
        """
        Load assets for a digital twin scene in

        """

    def _build_actor_helper(
        self,
        model_id: str,
        scale: float = 1.0,
        physical_material: physx.PhysxMaterial = None,
        density: float = 1000.0,
        root_dir: str = ASSET_DIR / "custom",
    ):
        builder = self.scene.create_actor_builder()
        model_dir = Path(root_dir) / "models" / model_id

        collision_file = str(model_dir / "collision.obj")
        builder.add_multiple_convex_collisions_from_file(
            filename=collision_file,
            scale=[scale] * 3,
            material=physical_material,
            density=density,
        )

        visual_file = str(model_dir / "textured.obj")
        if not os.path.exists(visual_file):
            visual_file = str(model_dir / "textured.dae")
            if not os.path.exists(visual_file):
                visual_file = str(model_dir / "textured.glb")
        builder.add_visual_from_file(filename=visual_file, scale=[scale] * 3)

        actor = builder.build(name=model_id)
        return actor

    def _after_reconfigure(self, options):
        target_object_actor_ids = [
            x._objs[0].per_scene_id
            for x in self.scene.actors.values()
            if x.name
            not in ["ground", "goal_site", "", "arena"]
            + self.rgb_always_overlay_objects
        ]
        self.target_object_actor_ids = torch.tensor(
            target_object_actor_ids, dtype=torch.int16, device=self.device
        )
        # get the robot link ids
        robot_links = (
            self.agent.robot.get_links()
        )  # e.g., [Actor(name="root", id="1"), Actor(name="root_arm_1_link_1", id="2"), Actor(name="root_arm_1_link_2", id="3"), ...]
        self.robot_link_ids = torch.tensor(
            [x._objs[0].entity.per_scene_id for x in robot_links],
            dtype=torch.int16,
            device=self.device,
        )

        # get the link ids of other articulated objects
        other_link_ids = []
        # for art_obj in self._scene.get_all_articulations():
        #     if art_obj is self.agent.robot:
        #         continue
        #     if art_obj.name in self.rgb_always_overlay_objects:
        #         continue
        #     for link in art_obj.get_links():
        #         other_link_ids.append(link.id)
        other_link_ids = np.array(other_link_ids, dtype=np.int32)

        self.rgb_overlay_images: dict[str, torch.Tensor] = dict()
        for camera_name in self.rgb_overlay_cameras:
            sensor = self._sensor_configs[camera_name]
            if isinstance(sensor, CameraConfig):
                rgb_overlay_img = cv2.resize(
                    self.rgb_overlay_img, (sensor.width, sensor.height)
                )
                self.rgb_overlay_images[camera_name] = common.to_tensor(rgb_overlay_img)

    def _green_sceen_rgb(self, rgb, segmentation, overlay_img):
        """returns green screened RGB data"""
        actor_seg = segmentation[..., 0]
        mask = torch.ones_like(actor_seg)
        if ("background" in self.rgb_overlay_mode) or (
            "debug" in self.rgb_overlay_mode
        ):
            if ("object" not in self.rgb_overlay_mode) or (
                "debug" in self.rgb_overlay_mode
            ):
                # only overlay the background and keep the foregrounds (robot and target objects) rendered in simulation
                mask[
                    torch.isin(
                        actor_seg,
                        torch.concatenate(
                            [self.robot_link_ids, self.target_object_actor_ids]
                        ),
                    )
                ] = 0
                # mask[np.isin(actor_seg, np.concatenate([robot_link_ids, target_object_actor_ids, other_link_ids]))] = 0.0
            else:
                # overlay everything except the robot links
                mask[np.isin(actor_seg, self.robot_link_ids)] = 0.0
        else:
            raise NotImplementedError(self.rgb_overlay_mode)
        mask = mask[..., None]

        # perform overlay on the RGB observation image
        if "debug" not in self.rgb_overlay_mode:
            rgb = rgb * (1 - mask) + overlay_img * mask
        else:
            # debug
            # obs['sensor_data'][camera_name]['Color'][..., :3] = obs['sensor_data'][camera_name]['Color'][..., :3] * (1 - mask) + rgb_overlay_img * mask
            rgb = rgb * 0.5 + overlay_img * 0.5
        return rgb

    def get_obs(self, info: dict = None):
        obs = super().get_obs(info)

        # "greenscreen" process
        if self._obs_mode == "rgb" and self.rgb_overlay_img is not None:
            # TODO (parallelize this?)
            # get the actor ids of objects to manipulate; note that objects here are not articulated

            for camera_name in self.rgb_overlay_cameras:
                # obtain overlay mask based on segmentation info
                assert (
                    "segmentation" in obs["sensor_data"][camera_name].keys()
                ), "Image overlay requires segment info in the observation!"
                green_screened_rgb = self._green_sceen_rgb(
                    obs["sensor_data"][camera_name]["rgb"],
                    obs["sensor_data"][camera_name]["segmentation"],
                    self.rgb_overlay_images[camera_name],
                )
                obs["sensor_data"][camera_name]["rgb"] = green_screened_rgb
        return obs
