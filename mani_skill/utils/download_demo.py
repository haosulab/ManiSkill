import argparse
import os
import os.path as osp
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from tqdm import tqdm

from mani_skill import DEMO_DIR


@dataclass
class DemoDatasetSource:
    raw_dataset_url: str
    """URL pointing to the raw dataset which does not contain any observations, just env states, actions, and reset kwargs"""
    pre_processed_dataset_url: Optional[str] = None
    """URL pointing to preprocessed versions if any"""
    env_type: str = "rigid_body"  # or soft_body


DATASET_SOURCES: dict[str, DemoDatasetSource] = {}

# Rigid body envs
for env_id in [
    "AnymalC-Reach-v1",
    "DrawTriangle-v1",
    "LiftPegUpright-v1",
    "PegInsertionSide-v1",
    "PickCube-v1",
    "PlugCharger-v1",
    "PokeCube-v1",
    "PullCube-v1",
    "PullCubeTool-v1",
    "PushCube-v1",
    "PushT-v1",
    "RollBall-v1",
    "StackCube-v1",
    "TwoRobotPickCube-v1",
    "TwoRobotStackCube-v1",
]:
    DATASET_SOURCES[env_id] = DemoDatasetSource(
        raw_dataset_url=f"https://huggingface.co/datasets/haosulab/ManiSkill_Demonstrations/resolve/main/demos/{env_id}.zip?download=true"
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


def download_file(base_path, url, verbose=True):
    filename = os.path.basename(urlparse(url).path)
    local_path = os.path.join(base_path, filename)
    tmp_local_path = os.path.join(base_path, filename + ".tmp")
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    if verbose:
        with tqdm(
            unit_scale=True,
        ) as t:
            urllib.request.urlretrieve(url, tmp_local_path, reporthook=tqdmhook(t))
    else:
        urllib.request.urlretrieve(url, tmp_local_path, reporthook=tqdmhook(t))

    os.rename(tmp_local_path, local_path)
    return local_path


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "uid",
        type=str,
        nargs="?",
        default="",
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
        print("Available dataset UIDs:")
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
        output_dir = str(DEMO_DIR)
        if args.output_dir:
            output_dir = args.output_dir
        if verbose:
            print(
                f"Downloading demonstrations to {osp.abspath(output_dir)} - {i+1}/{len(uids)}, {uid}"
            )
        local_path = download_file(
            output_dir,
            meta.raw_dataset_url,
        )
        if osp.splitext(local_path)[1] == ".zip":
            with zipfile.ZipFile(local_path, "r") as zip_ref:
                zip_ref.extractall(output_dir)
            os.remove(local_path)


if __name__ == "__main__":
    main(parse_args())
