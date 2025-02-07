import copy
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
import h5py
import numpy as np
import tyro
from tqdm import tqdm

import mani_skill.envs
from mani_skill.trajectory import utils
from mani_skill.utils.visualization.misc import images_to_video


@dataclass
class Args:
    runs_path: str
    out_dir: str
    dry_run: Optional[bool] = True


def main():
    args = tyro.cli(Args)

    if args.dry_run:
        print("Dry run, skipping actual processing")

    # Dictionary to store paths for each environment experiment
    env_paths = {}

    # List all subfolders in runs_path
    for env_name in os.listdir(args.runs_path):
        env_dir = Path(args.runs_path) / env_name

        if not env_dir.is_dir():
            continue

        # Look for checkpoint and trajectory files
        ckpt_path = env_dir / "final_ckpt.pt"
        traj_path = env_dir / "test_videos" / "trajectory.h5"
        traj_metadata_path = env_dir / "test_videos" / "trajectory.json"

        # Only store if both files exist
        if not (ckpt_path.exists() and traj_path.exists()):
            print(
                f"Skipping {env_name} because checkpoint or trajectory file does not exist"
            )
            continue

        env_paths[env_name] = {
            "checkpoint": str(ckpt_path),
            "trajectory": str(traj_path),
            "metadata": str(traj_metadata_path),
        }
    high_fail_rate_envs = []
    for env_name, env_path in env_paths.items():
        print(f"Processing {env_name}")
        traj_path = env_path["trajectory"]
        file = h5py.File(traj_path, "r")
        metadata_path = env_path["metadata"]
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        env_id = metadata["env_info"]["env_id"]
        new_metadata = copy.deepcopy(metadata)
        new_metadata["episodes"] = []
        control_mode = new_metadata["env_info"]["env_kwargs"]["control_mode"]
        sim_backend = new_metadata["env_info"]["env_kwargs"]["sim_backend"]
        traj_filename = f"{env_id}/rl/trajectory.none.{control_mode}.{sim_backend}"
        out_trajectory_path = os.path.join(args.out_dir, f"{traj_filename}.h5")

        if not args.dry_run:
            os.makedirs(os.path.dirname(out_trajectory_path), exist_ok=True)
            out_file = h5py.File(out_trajectory_path, "w")

        failed_count = 0
        truncated_count = 0
        avg_episode_length = 0
        original_episode_count = len(metadata["episodes"])
        first_success_indexes = []
        recorded_sample_video = False
        for episode in tqdm(metadata["episodes"]):
            traj_id = f"traj_{episode['episode_id']}"
            traj = file[traj_id]
            success = np.array(traj["success"])
            if not success.any():
                # this failed
                failed_count += 1
                continue
            # truncate until last success
            success_indexes = success.nonzero()[0]
            last_success_index = int(success_indexes[-1])
            first_success_index = int(success_indexes[0])
            first_success_indexes.append(first_success_index)
            if last_success_index != len(success) - 1:
                truncated_count += 1
            avg_episode_length += last_success_index + 1

            def recursive_copy_and_slice(
                key, source_group, target_group, add_last_frame=False
            ):
                if key == "obs" or key == "rewards":
                    return
                if isinstance(target_group, h5py.Dataset):
                    if not add_last_frame and ("obs" == key or "env_states" == key):
                        add_last_frame = True
                    source_group.create_dataset(
                        key,
                        data=target_group[: last_success_index + 1 + add_last_frame],
                    )
                elif isinstance(target_group, h5py.Group):
                    if not add_last_frame and ("obs" == key or "env_states" == key):
                        add_last_frame = True
                    source_group.create_group(key, track_order=True)
                    for k in target_group.keys():
                        recursive_copy_and_slice(
                            k,
                            source_group[key],
                            target_group[k],
                            add_last_frame=add_last_frame,
                        )

            if not args.dry_run:
                recursive_copy_and_slice(traj_id, out_file, traj)
            new_episode = copy.deepcopy(episode)
            new_episode["success"] = True
            new_episode["elapsed_steps"] = last_success_index + 1
            new_metadata["episodes"].append(new_episode)

            if not args.dry_run:
                if not recorded_sample_video:
                    recorded_sample_video = True
                    env_kwargs = copy.deepcopy(new_metadata["env_info"]["env_kwargs"])
                    env_kwargs["num_envs"] = 1
                    env_kwargs["sim_backend"] = "physx_cpu"
                    env_kwargs["human_render_camera_configs"] = {
                        "shader_pack": "rt-med"
                    }
                    env = gym.make(env_id, **env_kwargs)
                    env.reset(
                        seed=episode["episode_seed"], **new_episode["reset_kwargs"]
                    )
                    imgs = []
                    env_states = utils.dict_to_list_of_dicts(
                        out_file[traj_id]["env_states"]
                    )
                    for step in range(new_episode["elapsed_steps"]):
                        env.set_state_dict(env_states[step])
                        imgs.append(env.render_rgb_array().cpu().numpy()[0])
                    env.close()
                    images_to_video(
                        imgs,
                        output_dir=os.path.join(args.out_dir, env_id, "rl"),
                        video_name=f"sample_{control_mode}",
                        fps=30,
                    )
        final_episode_count = len(new_metadata["episodes"])
        avg_episode_length /= final_episode_count
        avg_steps_to_first_success = np.mean(first_success_indexes)
        print(
            f"{env_id}: Failed: {failed_count}/{original_episode_count}, Truncated: {truncated_count}/{original_episode_count}, Final Episodes: {final_episode_count}, Avg Episode Length: {avg_episode_length}, Avg Steps to First Success: {avg_steps_to_first_success}"
        )
        if failed_count / original_episode_count >= 0.05:
            high_fail_rate_envs.append(
                (env_name, failed_count / original_episode_count)
            )

        new_metadata["source_type"] = "rl"
        new_metadata[
            "source_desc"
        ] = "Demonstrations generated by rolling out a PPO dense reward trained policy"
        if not args.dry_run:
            with open(
                os.path.join(
                    args.out_dir,
                    f"{traj_filename}.json",
                ),
                "w",
            ) as f:
                json.dump(new_metadata, f, indent=2)
            print(f"Saved to {os.path.join(args.out_dir, f'{traj_filename}.json')}")
            out_file.close()

            # Copy checkpoint to output dir
            checkpoint_path = env_path["checkpoint"]
            checkpoint_out_path = os.path.join(
                args.out_dir, f"{env_id}/rl/ppo_{control_mode}_ckpt.pt"
            )
            os.makedirs(os.path.dirname(checkpoint_out_path), exist_ok=True)
            shutil.copy(checkpoint_path, checkpoint_out_path)

    for env_name, fail_rate in high_fail_rate_envs:
        print(
            f"Warning: {env_name} has {fail_rate*100:0.1f} >= 5% failed episodes. Need a better policy."
        )


if __name__ == "__main__":
    main()
