"""
Code for converting to the Open-X Embodiment dataset format using RLDS from ManiSkill datasets
"""

from typing import Any, Iterator, Tuple

import cv2
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

# TODO (stao): remove old comments/code and add new tasks where possible.


class ManiSkill3OpenXEmbodimentDataset(tfds.core.GeneratorBasedBuilder):
    """
    Example RLDS dataset builder in the Open-X Embodiment format for some ManiSkill tasks/demonstrations.

    Given the highly specific nature of Open-X Embodiment dataset format, there is currently not a general tool
    to convert to the specific format and strongly recommend you to use code as is or copy the code and modify
    for your own needs. In particular the code here currently supports converting ManiSkill demonstrations for
    the following tasks:
    - PickCube-v1
    - StackCube-v1
    - PlugCharger-v1
    - PegInsertionSide-v1

    to load this you need to download and preprocess existing demonstrations to generate the visual data and actions in the appropriate action space

    ```bash
    python -m mani_skill.utils.download_demo PickCube-v1 StackCube-v1 PlugCharger-v1 PegInsertionSide-v1
    ```

    ```bash
    for env_id in PickCube-v1 StackCube-v1 PlugCharger-v1 PegInsertionSide-v1; do
        python -m mani_skill.trajectory.replay_trajectory --traj-path ~/.maniskill/demos/${env_id}/motionplanning/trajectory.h5 \
            -c pd_ee_target_delta_pose -o rgbd --save-traj --count 100 --num-procs 10
    done
    ```


    The specific parts of Open-X embodiment that are hard-coded in this conversion tool are noted below:
    - There are two cameras, a main (base) camera and a wrist camera
    - Each camera has a 256x256 resolution and provides RGB+Depth data
    - Assumes the task's observation includes "tcp_pose" as a property in the "extra" field.
    - Has some hardcoded numbers per task that define what parts of observations are goal information, and what parts of goals are used for success checks
    """

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    NUM_EPISODES_PER_ENV = -1

    def __init__(
        self, data_root: str = "/home/stao/.maniskill/demos/", *args, **kwargs
    ):
        self.image_resolution = kwargs.pop("img_resolution", 256)
        super().__init__(*args, **kwargs)
        # self.data_path = {}
        self._embed = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        )
        self.data_root = data_root
        self.language_instruction_dict = {
            "PickCube-v1": "Pick up the red cube and move it to a goal position.",
            "StackCube-v1": "Stack the red cube on top of the green cube.",
            "PlugCharger-v1": "Plug the charger into the wall socket.",
            "PegInsertionSide-v1": "Insert the peg into the horizontal hole in a box.",
            # "LiftCube-v0": "Lift up the red cube by 0.2 meters.",
            # "PickCube-v0": "Pick up the red cube and move it to a goal position.",
            # "StackCube-v0": "Stack the red cube on top of the green cube.",
            # "PlugCharger-v0": "Plug the charger into the wall socket.",
            # "PegInsertionSide-v0": "Insert the peg into the horizontal hole in a box.",
            # "AssemblingKits-v0": "Insert a designated object into the corresponding slot on a board.",
            # "PickSingleYCB-v0": "Pick up the object and move it to a goal position.",
            # "PickSingleEGAD-v0": "Pick up the object and move it to a goal position.",
            # "PickClutterYCB-v0": "Pick up a designated object from a clutter of objects.",
            # "TurnFaucet-v0": "Turn on the faucet by rotating a designated handle.",
        }
        """language instruction for each task"""
        self.target_object_or_part_initial_pose_fn_dict = {
            "PickCube-v1": None,
            "StackCube-v1": None,
            "PlugCharger-v1": None,
            "PegInsertionSide-v1": None,
            # "LiftCube-v0": None,
            # "PickCube-v0": None,
            # "StackCube-v0": None,
            # "PlugCharger-v0": None,
            # "PegInsertionSide-v0": None,
            # "AssemblingKits-v0": lambda obs, i: np.concatenate(
            #     [obs["extra/obj_init_pos"][i], [1, 0, 0, 0]]
            # ),
            # "PickSingleYCB-v0": None,
            # "PickSingleEGAD-v0": None,
            # "PickClutterYCB-v0": lambda obs, i: np.concatenate(
            #     [obs["extra/obj_start_pos"][i], [1, 0, 0, 0]]
            # ),
            # "TurnFaucet-v0": lambda obs, i: np.concatenate(
            #     [obs["extra/target_link_pos"][i], [1, 0, 0, 0]]
            # ),
        }
        """???"""
        self.target_object_or_part_initial_pose_valid_dict = {
            "PickCube-v1": np.zeros(7, dtype=np.uint8),
            "StackCube-v1": np.zeros(7, dtype=np.uint8),
            "PlugCharger-v1": np.zeros(7, dtype=np.uint8),
            "PegInsertionSide-v1": np.zeros(7, dtype=np.uint8),
            # "LiftCube-v0": np.zeros(7, dtype=np.uint8),
            # "PickCube-v0": np.zeros(7, dtype=np.uint8),
            # "StackCube-v0": np.zeros(7, dtype=np.uint8),
            # "PlugCharger-v0": np.zeros(7, dtype=np.uint8),
            # "PegInsertionSide-v0": np.zeros(7, dtype=np.uint8),
            # "AssemblingKits-v0": np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.uint8),
            # "PickSingleYCB-v0": np.zeros(7, dtype=np.uint8),
            # "PickSingleEGAD-v0": np.zeros(7, dtype=np.uint8),
            # "PickClutterYCB-v0": np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.uint8),
            # "TurnFaucet-v0": np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.uint8),
        }
        """???"""
        self.target_object_or_part_final_pose_fn_dict = {
            "PickCube-v1": lambda obs, i: np.concatenate(
                [obs["extra/goal_pos"][i], [1, 0, 0, 0]]
            ),
            "StackCube-v1": None,
            "PlugCharger-v1": None,
            "PegInsertionSide-v1": None,
            # "LiftCube-v0": None,
            # "PickCube-v0": lambda obs, i: np.concatenate(
            #     [obs["extra/goal_pos"][i], [1, 0, 0, 0]]
            # ),
            # "StackCube-v0": None,
            # "PlugCharger-v0": None,
            # "PegInsertionSide-v0": None,
            # "AssemblingKits-v0": lambda obs, i: np.concatenate(
            #     [obs["extra/obj_goal_pos"][i], [1, 0, 0, 0]]
            # ),
            # "PickSingleYCB-v0": lambda obs, i: np.concatenate(
            #     [obs["extra/goal_pos"][i], [1, 0, 0, 0]]
            # ),
            # "PickSingleEGAD-v0": lambda obs, i: np.concatenate(
            #     [obs["extra/goal_pos"][i], [1, 0, 0, 0]]
            # ),
            # "PickClutterYCB-v0": lambda obs, i: np.concatenate(
            #     [obs["extra/goal_pos"][i], [1, 0, 0, 0]]
            # ),
            # "TurnFaucet-v0": lambda obs, i: np.concatenate(
            #     [
            #         obs["extra/target_link_pos"][i],
            #         [np.cos(obs["extra/target_angle_diff"][i])],
            #         np.sin(obs["extra/target_angle_diff"][i])
            #         * obs["extra/target_joint_axis"][i],
            #     ]
            # ),
        }
        """???"""
        self.target_object_or_part_final_pose_valid_dict = {
            "PickCube-v1": np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.uint8),
            "StackCube-v1": np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.uint8),
            "PlugCharger-v1": np.zeros(7, dtype=np.uint8),
            "PegInsertionSide-v1": np.zeros(7, dtype=np.uint8),
            # "LiftCube-v0": np.zeros(7, dtype=np.uint8),
            # "PickCube-v0": np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.uint8),
            # "StackCube-v0": np.zeros(7, dtype=np.uint8),
            # "PlugCharger-v0": np.zeros(7, dtype=np.uint8),
            # "PegInsertionSide-v0": np.zeros(7, dtype=np.uint8),
            # "AssemblingKits-v0": np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.uint8),
            # "PickSingleYCB-v0": np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.uint8),
            # "PickSingleEGAD-v0": np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.uint8),
            # "PickClutterYCB-v0": np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.uint8),
            # "TurnFaucet-v0": np.ones(7, dtype=np.uint8),
        }
        """???"""

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(
                                            self.image_resolution,
                                            self.image_resolution,
                                            3,
                                        ),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Main camera RGB observation.",
                                    ),
                                    "depth": tfds.features.Image(
                                        shape=(
                                            self.image_resolution,
                                            self.image_resolution,
                                            1,
                                        ),
                                        dtype=np.uint16,
                                        encoding_format="png",
                                        doc="Main camera Depth observation. Divide the depth value by 2**10 to get the depth in meters.",
                                    ),
                                    "wrist_image": tfds.features.Image(
                                        shape=(
                                            self.image_resolution,
                                            self.image_resolution,
                                            3,
                                        ),
                                        dtype=np.uint8,
                                        encoding_format="png",
                                        doc="Wrist camera RGB observation.",
                                    ),
                                    "wrist_depth": tfds.features.Image(
                                        shape=(
                                            self.image_resolution,
                                            self.image_resolution,
                                            1,
                                        ),
                                        dtype=np.uint16,
                                        encoding_format="png",
                                        doc="Wrist camera Depth observation. Divide the depth value by 2**10 to get the depth in meters.",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(18,),
                                        dtype=np.float32,
                                        doc="Robot state, consists of [7x robot joint angles, "
                                        "2x gripper position, 7x robot joint angle velocity, "
                                        "2x gripper velocity]. Angle in radians, position in meters.",
                                    ),
                                    # "base_pose": tfds.features.Tensor(
                                    #     shape=(7,),
                                    #     dtype=np.float32,
                                    #     doc="Robot base pose in the world frame, consists of [x, y, z, qw, qx, qy, qz]. "
                                    #     "The first three dimensions represent xyz positions in meters. "
                                    #     "The last four dimensions are the quaternion representation of rotation.",
                                    # ),
                                    "tcp_pose": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float32,
                                        doc="Robot tool-center-point pose in the world frame, consists of [x, y, z, qw, qx, qy, qz]. "
                                        "Tool-center-point is the center between the two gripper fingers.",
                                    ),
                                    "target_object_or_part_initial_pose": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float32,
                                        doc="The initial pose of the target object or object part to be manipulated, consists of [x, y, z, qw, qx, qy, qz]. "
                                        "The pose is represented in the world frame. "
                                        "This variable is used to specify the target object or object part when multiple objects or object parts are present in an environment",
                                    ),
                                    "target_object_or_part_initial_pose_valid": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.uint8,
                                        doc="Whether each dimension of target_object_or_part_initial_pose is valid in an environment. "
                                        "1 = valid; 0 = invalid (in which case one should ignore the corresponding dimensions in target_object_or_part_initial_pose).",
                                    ),
                                    "target_object_or_part_final_pose": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.float32,
                                        doc="The final pose towards which the target object or object part needs be manipulated, consists of [x, y, z, qw, qx, qy, qz]. "
                                        "The pose is represented in the world frame. "
                                        "An episode is considered successful if the target object or object part is manipulated to this pose.",
                                    ),
                                    "target_object_or_part_final_pose_valid": tfds.features.Tensor(
                                        shape=(7,),
                                        dtype=np.uint8,
                                        doc="Whether each dimension of target_object_or_part_final_pose is valid in an environment. "
                                        "1 = valid; 0 = invalid (in which case one should ignore the corresponding dimensions in target_object_or_part_final_pose). "
                                        '"Invalid" means that there is no success check on the final pose of target object or object part in the corresponding dimensions.',
                                    ),
                                    "main_camera_extrinsic_cv": tfds.features.Tensor(
                                        shape=(3, 4),
                                        dtype=np.float32,
                                        doc="Main camera extrinsic matrix in OpenCV convention.",
                                    ),
                                    "main_camera_intrinsic_cv": tfds.features.Tensor(
                                        shape=(3, 3),
                                        dtype=np.float32,
                                        doc="Main camera intrinsic matrix in OpenCV convention.",
                                    ),
                                    "main_camera_cam2world_gl": tfds.features.Tensor(
                                        shape=(4, 4),
                                        dtype=np.float32,
                                        doc="Transformation from the main camera frame to the world frame in OpenGL/Blender convention.",
                                    ),
                                    "wrist_camera_extrinsic_cv": tfds.features.Tensor(
                                        shape=(3, 4),
                                        dtype=np.float32,
                                        doc="Wrist camera extrinsic matrix in OpenCV convention.",
                                    ),
                                    "wrist_camera_intrinsic_cv": tfds.features.Tensor(
                                        shape=(3, 3),
                                        dtype=np.float32,
                                        doc="Wrist camera intrinsic matrix in OpenCV convention.",
                                    ),
                                    "wrist_camera_cam2world_gl": tfds.features.Tensor(
                                        shape=(4, 4),
                                        dtype=np.float32,
                                        doc="Transformation from the wrist camera frame to the world frame in OpenGL/Blender convention.",
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(7,),
                                dtype=np.float32,
                                doc="Robot action, consists of [3x end effector delta target position, "
                                "3x end effector delta target orientation in axis-angle format, "
                                "1x gripper target position (mimic for two fingers)]. "
                                "For delta target position, an action of -1 maps to a robot movement of -0.1m, and action of 1 maps to a movement of 0.1m. "
                                "For delta target orientation, its encoded angle is mapped to a range of [-0.1rad, 0.1rad] for robot execution. "
                                "For example, an action of [1, 0, 0] means rotating along the x-axis by 0.1 rad. "
                                "For gripper target position, an action of -1 means close, and an action of 1 means open.",
                            ),
                            "discount": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Discount if provided, default to 1.",
                            ),
                            "reward": tfds.features.Scalar(
                                dtype=np.float32,
                                doc="Reward if provided, 1 on final step for demos.",
                            ),
                            "is_first": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on first step of the episode."
                            ),
                            "is_last": tfds.features.Scalar(
                                dtype=np.bool_, doc="True on last step of the episode."
                            ),
                            "is_terminal": tfds.features.Scalar(
                                dtype=np.bool_,
                                doc="True on last step of the episode if it is a terminal step, True for demos.",
                            ),
                            "language_instruction": tfds.features.Text(
                                doc="Language Instruction."
                            ),
                            # "language_embedding": tfds.features.Tensor(
                            #     shape=(512,),
                            #     dtype=np.float32,
                            #     doc="Kona language embedding. "
                            #     "See https://tfhub.dev/google/universal-sentence-encoder-large/5",
                            # ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(
                                doc="Path to the original data file."
                            ),
                            "episode_id": tfds.features.Text(doc="Episode ID."),
                        }
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        # data_root = self.data_path
        data_root = "/home/stao/.maniskill/demos/"
        paths = [
            "PickCube-v1/motionplanning/trajectory.rgbd.pd_ee_target_delta_pose.cpu.h5",
            "StackCube-v1/motionplanning/trajectory.rgbd.pd_ee_target_delta_pose.cpu.h5",
            "PlugCharger-v1/motionplanning/trajectory.rgbd.pd_ee_target_delta_pose.cpu.h5",
            "PegInsertionSide-v1/motionplanning/trajectory.rgbd.pd_ee_target_delta_pose.cpu.h5",
            # "LiftCube-v0/trajectory.rgbd.pd_base_ee_target_delta_pose.h5",
            # "PickCube-v0/trajectory.rgbd.pd_base_ee_target_delta_pose.h5",
            # "StackCube-v0/trajectory.rgbd.pd_base_ee_target_delta_pose.h5",
            # "PlugCharger-v0/trajectory.rgbd.pd_base_ee_target_delta_pose.h5",
            # "PegInsertionSide-v0/trajectory.rgbd.pd_base_ee_target_delta_pose.h5",
            # "AssemblingKits-v0/trajectory.rgbd.pd_base_ee_target_delta_pose.h5",
            # "PickSingleYCB-v0/trajectory_merged.rgbd.pd_base_ee_target_delta_pose.h5",
            # "PickSingleEGAD-v0/trajectory_merged.rgbd.pd_base_ee_target_delta_pose.h5",
            # "PickClutterYCB-v0/trajectory_merged.rgbd.pd_base_ee_target_delta_pose.h5",
            # "TurnFaucet-v0/trajectory_merged.rgbd.pd_base_ee_target_delta_pose.h5",
        ]
        env_names = [path.split("/")[0] for path in paths]
        paths = [data_root + path for path in paths]
        # return {
        #     env_name: self._generate_examples([env_name], [path])
        #     for env_name, path in zip(env_names, paths)
        # }
        return {
            # 'train': self._generate_examples(path='data/train/LiftCube-v0/trajectory.rgbd.pd_joint_pos.h5'),
            "train": self._generate_examples(env_names, paths),
        }

    def _generate_examples(self, env_names, paths) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(d):
            env_name = d["env_name"]
            h5_path = d["h5_path"]
            episode_id = d["episode_id"]
            language_instruction = self.language_instruction_dict[env_name]
            target_object_or_part_initial_pose_fn = (
                self.target_object_or_part_initial_pose_fn_dict[env_name]
            )
            target_object_or_part_initial_pose_valid = (
                self.target_object_or_part_initial_pose_valid_dict[env_name]
            )
            target_object_or_part_final_pose_fn = (
                self.target_object_or_part_final_pose_fn_dict[env_name]
            )
            target_object_or_part_final_pose_valid = (
                self.target_object_or_part_final_pose_valid_dict[env_name]
            )

            h5_file = h5py.File(h5_path, "r")

            h5_episode = h5_file[episode_id]
            obs = h5_episode["obs"]
            actions = h5_episode["actions"]

            # language_embedding = self._embed([language_instruction])[0].numpy()

            episode = []
            ep_len = actions.shape[0] + 1

            for i in range(ep_len):
                base_image = obs["sensor_data"]["base_camera"]["rgb"][i]
                base_depth = obs["sensor_data"]["base_camera"]["depth"][i]
                if "hand_camera" in obs["sensor_data"]:
                    wrist_image = obs["sensor_data"]["hand_camera"]["rgb"][i]
                    wrist_depth = obs["sensor_data"]["hand_camera"]["depth"][i]
                else:
                    wrist_image = np.zeros((256, 256, 3), dtype=np.uint8)
                    wrist_depth = np.zeros((256, 256, 1), dtype=np.uint16)
                qpos = obs["agent"]["qpos"][i]
                qvel = obs["agent"]["qvel"][i]
                tcp_pose = obs["extra"]["tcp_pose"][i]

                main_camera_extrinsic_cv = obs["sensor_param"]["base_camera"][
                    "extrinsic_cv"
                ][i]
                main_camera_intrinsic_cv = obs["sensor_param"]["base_camera"][
                    "intrinsic_cv"
                ][i]
                main_camera_cam2world_gl = obs["sensor_param"]["base_camera"][
                    "cam2world_gl"
                ][i]
                if "hand_camera" in obs["sensor_param"]:
                    wrist_camera_extrinsic_cv = obs["sensor_param"]["hand_camera"][
                        "extrinsic_cv"
                    ][i]
                    wrist_camera_intrinsic_cv = obs["sensor_param"]["hand_camera"][
                        "intrinsic_cv"
                    ][i]
                    wrist_camera_cam2world_gl = obs["sensor_param"]["hand_camera"][
                        "cam2world_gl"
                    ][i]
                else:
                    wrist_camera_extrinsic_cv = np.zeros((3, 4), dtype=np.float32)
                    wrist_camera_intrinsic_cv = np.zeros((3, 3), dtype=np.float32)
                    wrist_camera_cam2world_gl = np.zeros((4, 4), dtype=np.float32)

                if target_object_or_part_initial_pose_fn is not None:
                    target_object_or_part_initial_pose = (
                        target_object_or_part_initial_pose_fn(obs, i).astype(np.float32)
                    )  # [7]
                else:
                    target_object_or_part_initial_pose = np.zeros(7, dtype=np.float32)
                if target_object_or_part_final_pose_fn is not None:
                    target_object_or_part_final_pose = (
                        target_object_or_part_final_pose_fn(obs, i).astype(np.float32)
                    )  # [7]
                else:
                    target_object_or_part_final_pose = np.zeros(7, dtype=np.float32)

                if i < ep_len - 1:
                    action = actions[i]  # [7]
                else:
                    action = np.zeros(7, dtype=np.float32)
                episode.append(
                    {
                        "observation": {
                            "image": base_image,
                            "depth": base_depth,
                            "wrist_image": wrist_image,
                            "wrist_depth": wrist_depth,
                            "state": np.concatenate([qpos, qvel]),
                            # "base_pose": base_pose,
                            "tcp_pose": tcp_pose,
                            "target_object_or_part_initial_pose": target_object_or_part_initial_pose,
                            "target_object_or_part_initial_pose_valid": target_object_or_part_initial_pose_valid,
                            "target_object_or_part_final_pose": target_object_or_part_final_pose,
                            "target_object_or_part_final_pose_valid": target_object_or_part_final_pose_valid,
                            "main_camera_extrinsic_cv": main_camera_extrinsic_cv,
                            "main_camera_intrinsic_cv": main_camera_intrinsic_cv,
                            "main_camera_cam2world_gl": main_camera_cam2world_gl,
                            "wrist_camera_extrinsic_cv": wrist_camera_extrinsic_cv,
                            "wrist_camera_intrinsic_cv": wrist_camera_intrinsic_cv,
                            "wrist_camera_cam2world_gl": wrist_camera_cam2world_gl,
                        },
                        "action": action,
                        "discount": 1.0,
                        "reward": float(i == (ep_len - 1)),
                        "is_first": i == 0,
                        "is_last": i == (ep_len - 1),
                        "is_terminal": i == (ep_len - 1),
                        "language_instruction": language_instruction,
                        # "language_embedding": language_embedding,
                    }
                )

            h5_file.close()

            # create output data sample
            sample = {
                "steps": episode,
                "episode_metadata": {
                    "file_path": h5_path,
                    "episode_id": "_".join([env_name, episode_id]),
                },
            }

            # if you want to skip an example for whatever reason, simply return None
            return "_".join([env_name, episode_id]), sample

        for env_name, path in zip(env_names, paths):
            with h5py.File(path, "r") as h5_file:
                episode_ids = sorted(h5_file.keys())
            if self.NUM_EPISODES_PER_ENV > 0:
                np.random.shuffle(episode_ids)
                episode_ids = episode_ids[: self.NUM_EPISODES_PER_ENV]
            print(env_name, "length", len(episode_ids))
            for episode_id in episode_ids:
                yield _parse_example(
                    {"env_name": env_name, "h5_path": path, "episode_id": episode_id}
                )
