import os
from typing import Dict, List

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

    This is based on the [SIMPLER](https://simpler-env.github.io/) and currently has the following tricks for
    making accurate simulated environments of real world datasets

    Greenscreening: Add a greenscreened real image to the background to make the images more realistic and more closer to the distribution
    of real world data.

    Note that this is not a general purpose system for building digital twins you can train and then transfer
    to the real world. This is designed to support fast evaluation in simulation of real world policies.
    """

    rgb_overlay_paths: Dict[str, str] = None
    """dict mapping camera name to the file path of the greenscreening image"""
    _rgb_overlay_images: Dict[str, torch.Tensor] = dict()
    rgb_always_overlay_objects: List[str] = []
    """List of names of actors/links that should be covered by the greenscreen"""
    rgb_overlay_mode: str = (
        "background"  # 'background' or 'object' or 'debug' or combinations of them
    )
    """which RGB overlay mode to use during the greenscreen process"""

    def __init__(self, **kwargs):
        # Load the "greenscreen" image, which is used to overlay the background portions of simulation observation
        if self.rgb_overlay_paths is not None:
            for camera_name, path in self.rgb_overlay_paths.items():
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"rgb_overlay_path {path} is not found."
                        "If you installed this repo through 'pip install .' , "
                        "you can download this directory https://github.com/simpler-env/ManiSkill2_real2sim/tree/main/data to get the real-world image overlay assets. "
                    )
                self._rgb_overlay_images[camera_name] = cv2.cvtColor(
                    cv2.imread(path), cv2.COLOR_BGR2RGB
                )  # (H, W, 3); float32
        else:
            self._rgb_overlay_images = None

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

    def _after_reconfigure(self, options: dict):
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
        robot_links = self.agent.robot.get_links()
        self.robot_link_ids = torch.tensor(
            [x._objs[0].entity.per_scene_id for x in robot_links],
            dtype=torch.int16,
            device=self.device,
        )

        for camera_name in self.rgb_overlay_paths.keys():
            sensor = self._sensor_configs[camera_name]
            if isinstance(sensor, CameraConfig):
                if isinstance(self._rgb_overlay_images[camera_name], torch.Tensor):
                    continue
                rgb_overlay_img = cv2.resize(
                    self._rgb_overlay_images[camera_name], (sensor.width, sensor.height)
                )
                self._rgb_overlay_images[camera_name] = common.to_tensor(
                    rgb_overlay_img, device=self.device
                )

    def _green_sceen_rgb(self, rgb, segmentation, overlay_img):
        """returns green screened RGB data given a batch of RGB and segmentation images and one overlay image"""
        actor_seg = segmentation[..., 0]
        mask = torch.ones_like(actor_seg, device=actor_seg.device)
        if actor_seg.device != self.robot_link_ids.device:
            # if using CPU simulation, the device of the robot_link_ids and target_object_actor_ids will be CPU first
            # but for most users who use the sapien_cuda render backend image data will be on the GPU.
            self.robot_link_ids = self.robot_link_ids.to(actor_seg.device)
            self.target_object_actor_ids = self.target_object_actor_ids.to(
                actor_seg.device
            )
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
            rgb = rgb * 0.5 + overlay_img * 0.5
        return rgb

    def get_obs(self, info: dict = None):
        obs = super().get_obs(info)

        # "greenscreen" process
        if (
            self.obs_mode_struct.visual.rgb
            and self.obs_mode_struct.visual.segmentation
            and self.rgb_overlay_paths is not None
        ):
            # get the actor ids of objects to manipulate; note that objects here are not articulated
            for camera_name in self._rgb_overlay_images.keys():
                # obtain overlay mask based on segmentation info
                assert (
                    "segmentation" in obs["sensor_data"][camera_name].keys()
                ), "Image overlay requires segment info in the observation!"
                if (
                    self._rgb_overlay_images[camera_name].device
                    != obs["sensor_data"][camera_name]["rgb"].device
                ):
                    self._rgb_overlay_images[camera_name] = self._rgb_overlay_images[
                        camera_name
                    ].to(obs["sensor_data"][camera_name]["rgb"].device)
                overlay_img = self._rgb_overlay_images[camera_name]
                green_screened_rgb = self._green_sceen_rgb(
                    obs["sensor_data"][camera_name]["rgb"],
                    obs["sensor_data"][camera_name]["segmentation"],
                    overlay_img,
                )
                obs["sensor_data"][camera_name]["rgb"] = green_screened_rgb
        return obs
