import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval.base_env import (
    BaseBridgeEnv,
)
from mani_skill.utils.registration import register_env


@register_env("PutCarrotOnPlateInScene-v1", max_episode_steps=60)
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

    def evaluate(self):
        info = super()._evaluate(
            success_require_src_completely_on_target=True,
        )
        for x in ["all_obj_keep_height", "near_tgt_obj", "is_closest_to_tgt"]:
            del info[x]
        return info

    def get_language_instruction(self, **kwargs):
        return "put carrot on plate"


@register_env("PutEggplantInBasketScene-v1", max_episode_steps=120)
class PutEggplantInBasketScene(BaseBridgeEnv):
    scene_setting = "sink"
    rgb_always_overlay_objects = ["sink", "dummy_sink_target_plane"]

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
                grid_pos.append(np.array([x + xy_center[0], y + xy_center[1], 0.936]))
        xyz_configs = [np.stack([pos, target_xy], axis=0) for pos in grid_pos]
        xyz_configs = torch.tensor(np.stack(xyz_configs))
        # Pose([-0.106847, 0.184165, 0.937616], [0.449144, 0.723185, 0.419872, 0.314612])
        quat_configs = torch.tensor(
            np.stack(
                [
                    np.array([euler2quat(0, 0, 0, "sxyz"), [1, 0, 0, 0]]),
                    np.array([euler2quat(0, 0, 1 * np.pi / 4, "sxyz"), [1, 0, 0, 0]]),
                    np.array([euler2quat(0, 0, -1 * np.pi / 4, "sxyz"), [1, 0, 0, 0]]),
                ]
            )
        )
        # print(xyz_configs)
        # Pose([-0.127357, 0.235652, 0.936485], [0.533208, 0.817232, -0.209365, -0.0631377])
        quat_configs = torch.tensor(
            [
                [[0.543729, 0.82549, 0.0746101, 0.131747], [1, 0, 0, 0]],
                # [[0.449144, 0.723185, 0.419872, 0.314612], [1, 0, 0, 0]],
                [[0.559342, 0.817133, -0.138906, -0.0116353], [1, 0, 0, 0]],
                [[0.543029, 0.789388, -0.267736, -0.101503], [1, 0, 0, 0]],
            ]
        )
        # Pose([-0.121931, 0.200983, 0.94], [0.543029, 0.789388, -0.267736, -0.101503])
        # Pose([-0.111394, 0.211813, 0.936457], [0.285242, 0.909909, -0.296087, -0.0550989])
        # Pose([-0.117329, 0.212145, 0.943674], [-0.0160361, 0.950431, -0.310455, 0.00642143])

        # Pose([-0.113776, 0.191183, 0.937244], )
        super().__init__(
            obj_names=[source_obj_name, target_obj_name],
            xyz_configs=xyz_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def evaluate(self, *args, **kwargs):
        return super()._evaluate(
            success_require_src_completely_on_target=False,
            z_flag_required_offset=0.06,
            *args,
            **kwargs,
        )

    def get_language_instruction(self, **kwargs):
        return "put eggplant into yellow basket"

    def _load_lighting(self, options):
        self.enable_shadow

        self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light(
            [0, 0, -1],
            [0.3, 0.3, 0.3],
            position=[0, 0, 1],
            shadow=False,
            shadow_scale=5,
            shadow_map_size=2048,
        )


@register_env("StackGreenCubeOnYellowCubeInScene-v1", max_episode_steps=60)
class StackGreenCubeOnYellowCubeInScene(BaseBridgeEnv):
    MODEL_JSON = "info_bridge_custom_baked_tex_v0.json"

    def __init__(
        self,
        **kwargs,
    ):
        xy_center = np.array([-0.16, 0.00])
        half_edge_length_xs = [0.05, 0.1]
        half_edge_length_ys = [0.05, 0.1]
        xyz_configs = []

        for (half_edge_length_x, half_edge_length_y) in zip(
            half_edge_length_xs, half_edge_length_ys
        ):
            grid_pos = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) * 2 - 1
            grid_pos = (
                grid_pos * np.array([half_edge_length_x, half_edge_length_y])[None]
                + xy_center[None]
            )

            for i, grid_pos_1 in enumerate(grid_pos):
                for j, grid_pos_2 in enumerate(grid_pos):
                    if i != j:
                        xyz_configs.append(
                            np.array(
                                [
                                    np.append(grid_pos_1, 0.887529),
                                    np.append(grid_pos_2, 0.887529),
                                ]
                            )
                        )

        quat_configs = [np.array([[1, 0, 0, 0], [1, 0, 0, 0]])]
        quat_configs = torch.tensor(quat_configs)
        xyz_configs = torch.tensor(np.stack(xyz_configs))
        source_obj_name = "baked_green_cube_3cm"
        target_obj_name = "baked_yellow_cube_3cm"
        super().__init__(
            obj_names=[source_obj_name, target_obj_name],
            xyz_configs=xyz_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def evaluate(self):
        info = super()._evaluate(
            success_require_src_completely_on_target=True,
        )
        for x in ["all_obj_keep_height", "near_tgt_obj", "is_closest_to_tgt"]:
            del info[x]
        return info

    def get_language_instruction(self, **kwargs):
        return "stack the green block on the yellow block"


@register_env("PutSpoonOnTableClothInScene-v1", max_episode_steps=60)
class PutSpoonOnTableClothInScene(BaseBridgeEnv):
    def __init__(
        self,
        **kwargs,
    ):
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
                            [np.append(grid_pos_1, 0.88), np.append(grid_pos_2, 0.875)]
                        )
                    )

        quat_configs = [
            np.array([[1, 0, 0, 0], [1, 0, 0, 0]]),
            np.array([euler2quat(0, 0, np.pi / 2), [1, 0, 0, 0]]),
        ]
        quat_configs = torch.tensor(quat_configs)
        xyz_configs = torch.tensor(np.stack(xyz_configs))
        source_obj_name = "bridge_spoon_generated_modified"
        target_obj_name = "table_cloth_generated_shorter"
        super().__init__(
            obj_names=[source_obj_name, target_obj_name],
            xyz_configs=xyz_configs,
            quat_configs=quat_configs,
            **kwargs,
        )

    def evaluate(self, success_require_src_completely_on_target=False, **kwargs):
        # this environment allows spoons to be partially on the table cloth to be considered successful
        return super()._evaluate(success_require_src_completely_on_target, **kwargs)

    def get_language_instruction(self, **kwargs):
        return "put the spoon on the towel"
