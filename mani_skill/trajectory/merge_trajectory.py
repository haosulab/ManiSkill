import argparse
from pathlib import Path

import h5py

from mani_skill.utils.io_utils import dump_json, load_json


def merge_h5(output_path: str, traj_paths, recompute_id=True):
    print("Merge to", output_path)

    merged_h5_file = h5py.File(output_path, "w")
    merged_json_path = output_path.replace(".h5", ".json")
    merged_json_data = {"env_info": {}, "episodes": []}
    _env_info = None
    cnt = 0

    for traj_path in traj_paths:
        traj_path = str(traj_path)
        print("Merging", traj_path)

        h5_file = h5py.File(traj_path, "r")
        json_path = traj_path.replace(".h5", ".json")
        json_data = load_json(json_path)

        # Check env info
        env_info = json_data["env_info"]
        if _env_info is None:
            _env_info = env_info
            merged_json_data["env_info"] = _env_info
        else:
            assert str(env_info) == str(_env_info), traj_path

        # Merge
        for ep in json_data["episodes"]:
            episode_id = ep["episode_id"]
            traj_id = f"traj_{episode_id}"

            # Copy h5 data
            if recompute_id:
                new_traj_id = f"traj_{cnt}"
            else:
                new_traj_id = traj_id

            assert new_traj_id not in merged_h5_file, new_traj_id
            h5_file.copy(traj_id, merged_h5_file, new_traj_id)

            # Copy json data
            if recompute_id:
                ep["episode_id"] = cnt
            merged_json_data["episodes"].append(ep)

            cnt += 1

        h5_file.close()

    # Ignore commit info
    merged_h5_file.close()
    dump_json(merged_json_path, merged_json_data, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dirs", nargs="+")
    parser.add_argument("-o", "--output-path", type=str)
    parser.add_argument("-p", "--pattern", type=str, default="trajectory.h5")
    args = parser.parse_args()

    traj_paths = []
    for input_dir in args.input_dirs:
        input_dir = Path(input_dir)
        traj_paths.extend(sorted(input_dir.rglob(args.pattern)))

    output_dir = Path(args.output_path).parent
    output_dir.mkdir(exist_ok=True, parents=True)

    merge_h5(args.output_path, traj_paths)


if __name__ == "__main__":
    main()
