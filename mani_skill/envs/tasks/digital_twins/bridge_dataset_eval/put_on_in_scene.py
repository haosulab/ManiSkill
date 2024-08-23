import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval.base_env import (
    BaseBridgeEnv,
)
from mani_skill.utils.registration import register_env


@register_env("PutCarrotOnPlateInScene-v0", max_episode_steps=60)
class PutCarrotOnPlateInScene(BaseBridgeEnv):
    scene_setting = "flat_table"

    def __init__(self, **kwargs):
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_x = 0.075
        half_edge_length_y = 0.075
        grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
        grid_pos = (
            grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None]
            + xy_center[None]
        )

        xyz_configs = []
        for i, grid_pos_1 in enumerate(grid_pos):
            for j, grid_pos_2 in enumerate(grid_pos):
                if i != j:
                    xyz_configs.append(
                        np.array(
                            [
                                np.append(grid_pos_1, 0.887529),
                                np.append(grid_pos_2, 0.869532),
                            ]
                        )
                    )
        xyz_configs = torch.tensor(np.stack(xyz_configs))
        quat_configs = torch.tensor(
            np.stack(
                [
                    np.array([euler2quat(0, 0, np.pi), [1, 0, 0, 0]]),
                    np.array([euler2quat(0, 0, -np.pi / 2), [1, 0, 0, 0]]),
                ]
            )
        )
        source_obj_name = "bridge_carrot_generated_modified"
        target_obj_name = "bridge_plate_objaverse_larger"
        super().__init__(
            obj_names=[source_obj_name, target_obj_name],
            xyz_configs=xyz_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def get_language_instruction(self, **kwargs):
        return "put carrot on plate"


@register_env("PutEggplantInBasketScene-v0", max_episode_steps=120)
class PutEggplantInBasketScene(BaseBridgeEnv):
    scene_setting = "sink"

    def __init__(self, **kwargs):
        source_obj_name = "eggplant"
        target_obj_name = "dummy_sink_target_plane"  # invisible

        target_xy = np.array([-0.125, 0.025, 1])
        xy_center = [-0.105, 0.206]

        half_span_x = 0.01
        half_span_y = 0.015
        num_x = 2
        num_y = 4

        grid_pos = []
        for x in np.linspace(-half_span_x, half_span_x, num_x):
            for y in np.linspace(-half_span_y, half_span_y, num_y):
                grid_pos.append(np.array([x + xy_center[0], y + xy_center[1], 0.888]))

        xyz_configs = [np.stack([pos, target_xy], axis=0) for pos in grid_pos]
        xyz_configs = torch.tensor(np.stack(xyz_configs))
        quat_configs = torch.tensor(
            np.stack(
                [
                    np.array([euler2quat(0, 0, 0, "sxyz"), [1, 0, 0, 0]]),
                    np.array([euler2quat(0, 0, 1 * np.pi / 4, "sxyz"), [1, 0, 0, 0]]),
                    np.array([euler2quat(0, 0, -1 * np.pi / 4, "sxyz"), [1, 0, 0, 0]]),
                ]
            )
        )
        super().__init__(
            obj_names=[source_obj_name, target_obj_name],
            xyz_configs=xyz_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def get_language_instruction(self, **kwargs):
        return "put carrot on plate"
