import argparse
import os
import os.path as osp
import urllib.request
import zipfile

from tqdm import tqdm

from mani_skill import DEMO_DIR

DATASET_SOURCES = {}

# Rigid body envs
DATASET_SOURCES["LiftCube-v0"] = dict(
    env_type="rigid_body",
    object_paths=[
        "rigid_body/LiftCube-v0/trajectory.h5",
        "rigid_body/LiftCube-v0/trajectory.json",
    ],
    latest_version=0,
)
DATASET_SOURCES["PickClutterYCB-v0"] = dict(
    env_type="rigid_body",
    object_paths=[
        "rigid_body/PickClutterYCB-v0/trajectory.h5",
        "rigid_body/PickClutterYCB-v0/trajectory.json",
    ],
    latest_version=0,
)
DATASET_SOURCES["AssemblingKits-v0"] = dict(
    env_type="rigid_body",
    object_paths=[
        "rigid_body/AssemblingKits-v0/trajectory.h5",
        "rigid_body/AssemblingKits-v0/trajectory.json",
    ],
    latest_version=0,
)
DATASET_SOURCES["TurnFaucet-v0"] = dict(
    env_type="rigid_body",
    object_paths=["rigid_body/TurnFaucet-v0/TurnFaucet-v0.zip"],
    latest_version=0,
)
DATASET_SOURCES["PandaAvoidObstacles-v0"] = dict(
    env_type="rigid_body",
    object_paths=[
        "rigid_body/PandaAvoidObstacles-v0/trajectory.h5",
        "rigid_body/PandaAvoidObstacles-v0/trajectory.json",
    ],
    latest_version=0,
)
DATASET_SOURCES["PickSingleYCB-v0"] = dict(
    env_type="rigid_body",
    object_paths=[
        "rigid_body/PickSingleYCB-v0/PickSingleYCB-v0.zip",
    ],
    latest_version=0,
)
DATASET_SOURCES["PlugCharger-v0"] = dict(
    env_type="rigid_body",
    object_paths=[
        "rigid_body/PlugCharger-v0/trajectory.h5",
        "rigid_body/PlugCharger-v0/trajectory.json",
    ],
    latest_version=0,
)
DATASET_SOURCES["PegInsertionSide-v0"] = dict(
    env_type="rigid_body",
    object_paths=[
        "rigid_body/PegInsertionSide-v0/trajectory.h5",
        "rigid_body/PegInsertionSide-v0/trajectory.json",
    ],
    latest_version=0,
)
DATASET_SOURCES["StackCube-v0"] = dict(
    env_type="rigid_body",
    object_paths=[
        "rigid_body/StackCube-v0/trajectory.h5",
        "rigid_body/StackCube-v0/trajectory.json",
    ],
    latest_version=0,
)
DATASET_SOURCES["PickCube-v0"] = dict(
    env_type="rigid_body",
    object_paths=[
        "rigid_body/PickCube-v0/trajectory.h5",
        "rigid_body/PickCube-v0/trajectory.json",
    ],
    latest_version=0,
)
DATASET_SOURCES["PushChair-v1"] = dict(
    env_type="rigid_body",
    object_paths=["rigid_body/PushChair-v1.zip"],
    latest_version=0,
)
DATASET_SOURCES["OpenCabinetDrawer-v1"] = dict(
    env_type="rigid_body",
    object_paths=["rigid_body/OpenCabinetDrawer-v1.zip"],
    latest_version=0,
)
DATASET_SOURCES["OpenCabinetDoor-v1"] = dict(
    env_type="rigid_body",
    object_paths=["rigid_body/OpenCabinetDoor-v1.zip"],
    latest_version=0,
)
DATASET_SOURCES["MoveBucket-v1"] = dict(
    env_type="rigid_body",
    object_paths=["rigid_body/MoveBucket-v1.zip"],
    latest_version=0,
)
# Soft body envs
DATASET_SOURCES["Write-v0"] = dict(
    env_type="soft_body",
    object_paths=[
        "soft_body/Write-v0/trajectory.h5",
        "soft_body/Write-v0/trajectory.json",
    ],
    latest_version=0,
)
DATASET_SOURCES["Pinch-v0"] = dict(
    env_type="soft_body",
    object_paths=[
        "soft_body/Pinch-v0/trajectory.h5",
        "soft_body/Pinch-v0/trajectory.json",
    ],
    latest_version=0,
)
DATASET_SOURCES["Hang-v0"] = dict(
    env_type="soft_body",
    object_paths=[
        "soft_body/Hang-v0/trajectory.h5",
        "soft_body/Hang-v0/trajectory.json",
    ],
    latest_version=0,
)
DATASET_SOURCES["Pour-v0"] = dict(
    env_type="soft_body",
    object_paths=[
        "soft_body/Pour-v0/trajectory.h5",
        "soft_body/Pour-v0/trajectory.json",
    ],
    latest_version=0,
)
DATASET_SOURCES["Excavate-v0"] = dict(
    env_type="soft_body",
    object_paths=[
        "soft_body/Excavate-v0/trajectory.h5",
        "soft_body/Excavate-v0/trajectory.json",
    ],
    latest_version=0,
)
DATASET_SOURCES["Fill-v0"] = dict(
    env_type="soft_body",
    object_paths=[
        "soft_body/Fill-v0/trajectory.h5",
        "soft_body/Fill-v0/trajectory.json",
    ],
    latest_version=0,
)

pbar = None


def tqdmhook(t):
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


def download_file(base_path, object_path, version=0, verbose=True):
    version_name = f"v{version}"
    local_path = os.path.join(base_path, version_name, object_path)
    tmp_local_path = os.path.join(base_path, version_name, object_path + ".tmp")
    object_path = os.path.join("demos", version_name, object_path)
    hf_url = f"https://huggingface.co/datasets/haosulab/ManiSkill2/resolve/main/{object_path}"
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)

    if verbose:
        with tqdm(
            unit_scale=True,
        ) as t:
            urllib.request.urlretrieve(hf_url, tmp_local_path, reporthook=tqdmhook(t))
    else:
        urllib.request.urlretrieve(hf_url, tmp_local_path, reporthook=tqdmhook(t))

    os.rename(tmp_local_path, local_path)
    return local_path


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "uid",
        type=str,
        help="An environment id (e.g. PickCube-v0), a type of environments (rigid_body/soft_body), or 'all' for all available demonstrations.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output.")
    parser.add_argument(
        "--download-version",
        type=int,
        help="Specify a specific version of the demonstrations to download by version number e.g. 2 means v2. If not specified (the default), the latest demo dataset version will always be downloaded",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="The directory to save demonstrations to. The files will then be saved to <output_dir>/<env_type>/<env_id>. By default it is saved to ~/.maniskill/demos or what MS_ASSET_DIR is set to.",
    )
    return parser.parse_args(args)


def main(args):
    verbose = not args.quiet
    if args.uid == "":
        print("Available uids:")
        print(list(DATASET_SOURCES.keys()))
        return

    if args.uid == "all":
        if verbose:
            print("All demonstrations will be downloaded. This may take a while.")
        uids = list(DATASET_SOURCES.keys())
    elif args.uid in ["rigid_body", "soft_body"]:
        uids = []
        for k, v in DATASET_SOURCES.items():
            if v["env_type"] == args.uid:
                uids.append(k)
    elif args.uid in DATASET_SOURCES:
        uids = [args.uid]
    else:
        raise KeyError("{} not found.".format(args.uid))

    for i, uid in enumerate(uids):
        meta = DATASET_SOURCES[uid]
        download_version = args.download_version
        if download_version is None:
            download_version = meta["latest_version"]
        elif download_version > meta["latest_version"]:
            raise ValueError(
                f"Version v{download_version} of demonstrations for {uid} do not exist. Latest version is v{meta['latest_version']}. If you think this is a bug please raise an issue on GitHub: https://github.com/haosulab/ManiSkill2/issues"
            )

        object_paths = meta["object_paths"]
        output_dir = str(DEMO_DIR)
        if args.output_dir:
            output_dir = args.output_dir
        final_path = osp.join(output_dir, f"v{download_version}", meta["env_type"])
        if verbose:
            print(
                f"Downloading v{download_version} demonstrations to {final_path} - {i+1}/{len(uids)}, {uid}"
            )
        for object_path in object_paths:
            local_path = download_file(
                output_dir, object_path, version=download_version
            )
            if osp.splitext(local_path)[1] == ".zip":
                with zipfile.ZipFile(local_path, "r") as zip_ref:
                    zip_ref.extractall(final_path)
                os.remove(local_path)


if __name__ == "__main__":
    main(parse_args())
