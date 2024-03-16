import argparse
import os
import os.path as osp
import urllib.request
import zipfile

from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm

from mani_skill import DEMO_DIR

DATASET_SOURCES = {}

# Rigid body envs
DATASET_SOURCES["PickCube-v1"] = dict(
    env_type="rigid_body",
    object_paths=[
        "PickCube-v1/teleop/0.mp4",
        "PickCube-v1/teleop/trajectory.h5",
        "PickCube-v1/teleop/trajectory.json",
    ],
)
DATASET_SOURCES["StackCube-v1"] = dict(
    env_type="rigid_body",
    object_paths=[
        "StackCube-v1/teleop/0.mp4",
        "StackCube-v1/teleop/trajectory.h5",
        "StackCube-v1/teleop/trajectory.json",
    ],
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


def download_file(base_path, object_path, verbose=True):
    local_path = os.path.join(base_path, object_path)
    tmp_local_path = os.path.join(base_path, object_path + ".tmp")
    object_path = os.path.join("demos", object_path)
    hf_url = (
        f"https://huggingface.co/datasets/haosulab/ManiSkill/resolve/main/{object_path}"
    )
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
        help="An environment id (e.g. PickCube-v1) or 'all' for all available demonstrations.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output.")
    # TODO: handle hugging face dataset git versioning here
    # parser.add_argument(
    #     "--download-version",
    #     type=int,
    #     help="Specify a specific version of the demonstrations to download by version number e.g. 2 means v2. If not specified (the default), the latest demo dataset version will always be downloaded",
    # )
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
        object_paths = meta["object_paths"]
        output_dir = str(DEMO_DIR)
        if args.output_dir:
            output_dir = args.output_dir
        final_path = osp.join(output_dir, uid)
        if verbose:
            print(
                f"Downloading demonstrations to {final_path} - {i+1}/{len(uids)}, {uid}"
            )
        for object_path in object_paths:
            local_path = download_file(
                output_dir,
                object_path,
            )
            if osp.splitext(local_path)[1] == ".zip":
                with zipfile.ZipFile(local_path, "r") as zip_ref:
                    zip_ref.extractall(final_path)
                os.remove(local_path)


if __name__ == "__main__":
    main(parse_args())
