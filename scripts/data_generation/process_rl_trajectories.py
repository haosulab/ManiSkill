from dataclasses import dataclass

import tyro


@dataclass
class Args:
    runs_path: str


def main():
    args = tyro.cli(Args)

    import os
    from pathlib import Path

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

        # Only store if both files exist
        assert ckpt_path.exists() and traj_path.exists()
        env_paths[env_name] = {
            "checkpoint": str(ckpt_path),
            "trajectory": str(traj_path),
        }
    print(env_paths)


if __name__ == "__main__":
    main()
