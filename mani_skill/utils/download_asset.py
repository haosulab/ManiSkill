import argparse
import hashlib
import os
import os.path as osp
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from urllib.error import URLError

from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

from mani_skill import ASSET_DIR, PACKAGE_ASSET_DIR
from mani_skill.utils.io_utils import load_json


@dataclass
class DataSource:
    source_type: str
    """what kind of data is this"""
    url: Optional[str] = None
    hf_repo_id: Optional[str] = None
    github_url: Optional[str] = None
    target_path: Optional[str] = None
    """the folder where the file will be downloaded to"""
    checksum: Optional[str] = None
    zip_dirname: Optional[str] = None
    """what to rename a zip files generated directory to"""
    filename: Optional[str] = None
    """name to change the downloaded file to. If None, will not change the name"""
    output_dir: str = ASSET_DIR


DATA_SOURCES: Dict[str, DataSource] = {}
DATA_GROUPS = {}


def initialize_sources():
    """
    Initialize the metadata for assets

    Note that the current organization works as follows

    - assets/* contain files for individual objects (.obj, .glb etc.) and articulations (.urdf etc.) that are generally reused.
        E.g. Partnet Mobility and the YCB dataset.
    - tasks/* contain files that are otherwise too big to upload to GitHub and are relevant for just that one task and will generally not be reused
    - scene_datasets/* is a bit messier but contains a self-contained folder for each scene dataset each of which is likely organized differently.
        These datasets often put their unique object and articulation files together with scene configuration data. In the future we will re-organize these
        datasets so that all objects are put into assets and leave scene_datasets for scene configuration information.
    - robots/* contains files for additional robots that are not included by default.

    """

    # TODO add google scanned objects
    DATA_SOURCES["ycb"] = DataSource(
        source_type="task_assets",
        url="https://huggingface.co/datasets/haosulab/ManiSkill2/resolve/main/data/mani_skill2_ycb.zip",
        target_path="assets/mani_skill2_ycb",
        checksum="174001ba1003cc0c5adda6453f4433f55ec7e804f0f0da22d015d525d02262fb",
    )
    DATA_GROUPS["PickSingleYCB-v1"] = ["ycb"]

    DATA_SOURCES["pick_clutter_ycb"] = DataSource(
        source_type="task_assets",
        url="https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/pick_clutter/ycb_train_5k.json.gz",
        target_path="tasks/pick_clutter",
        checksum="70ec176c7036f326ea7813b77f8c03bea9db5960198498957a49b2895a9ec338",
    )
    DATA_GROUPS["PickClutterYCB-v1"] = ["ycb", "pick_clutter_ycb"]

    DATA_SOURCES["assembling_kits"] = DataSource(
        source_type="task_assets",
        url="https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/assembling_kits_v1.zip",
        target_path="tasks/assembling_kits",
        checksum="e3371f17a07a012edaa3a0b3604fb1577f3fb921876c3d5ed59733dd75a6b4a0",
    )
    DATA_GROUPS["AssemblingKits-v1"] = ["assembling_kits"]

    DATA_SOURCES["panda_avoid_obstacles"] = DataSource(
        source_type="task_assets",
        url="https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/avoid_obstacles/panda_train_2k.json.gz",
        target_path="tasks/avoid_obstacles",
        checksum="44dae9a0804172515c290c1f49a1e7e72d76e40201a2c5c7d4a3ccd43b4d5be4",
    )
    DATA_GROUPS["PandaAvoidObstacles-v1"] = ["panda_avoid_obstacles"]

    DATA_SOURCES["pinch"] = DataSource(
        source_type="task_assets",
        url="https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/pinch.zip",
        target_path="tasks/pinch",
        checksum="3281d2d777fad42e6d37371b2d3ee16fb1c39984907176718ca2e4f447326fe7",
    )
    DATA_GROUPS["Pinch-v1"] = ["pinch"]

    DATA_SOURCES["write"] = DataSource(
        source_type="task_assets",
        url="https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/write.zip",
        target_path="tasks/write",
        checksum="c5b49e581bfed9cfb2107a607faf52795f840e93f5a7ad389290314513b4b634",
    )
    DATA_GROUPS["Write-v1"] = ["write"]

    # ---------------------------------------------------------------------------- #
    # PartNet-mobility
    # ---------------------------------------------------------------------------- #
    category_uids = {}
    for category in ["cabinet_drawer", "cabinet_door", "chair", "bucket", "faucet"]:
        model_json = (
            PACKAGE_ASSET_DIR / f"partnet_mobility/meta/info_{category}_train.json"
        )
        model_ids = set(load_json(model_json).keys())
        category_uids[category] = []
        for model_id in model_ids:
            uid = f"partnet_mobility/{model_id}"
            DATA_SOURCES[uid] = DataSource(
                source_type="objects",
                url=f"https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/partnet_mobility/dataset/{model_id}.zip",
                target_path=ASSET_DIR / "partnet_mobility" / "dataset" / model_id,
            )
            category_uids[category].append(uid)

    DATA_GROUPS["partnet_mobility_cabinet"] = set(
        category_uids["cabinet_drawer"] + category_uids["cabinet_door"]
    )
    DATA_GROUPS["partnet_mobility_chair"] = category_uids["chair"]
    DATA_GROUPS["partnet_mobility_bucket"] = category_uids["bucket"]
    DATA_GROUPS["partnet_mobility_faucet"] = category_uids["faucet"]
    DATA_GROUPS["partnet_mobility"] = set(
        category_uids["cabinet_drawer"]
        + category_uids["cabinet_door"]
        + category_uids["chair"]
        + category_uids["bucket"]
        + category_uids["faucet"]
    )

    DATA_GROUPS["OpenCabinetDrawer-v1"] = category_uids["cabinet_drawer"]
    DATA_GROUPS["OpenCabinetDoor-v1"] = category_uids["cabinet_door"]
    DATA_GROUPS["PushChair-v1"] = category_uids["chair"]
    DATA_GROUPS["MoveBucket-v1"] = category_uids["bucket"]
    DATA_GROUPS["TurnFaucet-v1"] = category_uids["faucet"]

    # ---------------------------------------------------------------------------- #
    # Interactable Scene Datasets
    # ---------------------------------------------------------------------------- #
    DATA_SOURCES["ReplicaCAD"] = DataSource(
        source_type="scene",
        hf_repo_id="haosulab/ReplicaCAD",
        target_path="scene_datasets/replica_cad_dataset",
    )

    DATA_SOURCES["ReplicaCADRearrange"] = DataSource(
        source_type="scene",
        url="https://huggingface.co/datasets/haosulab/ReplicaCADRearrange/resolve/main/v1_extracted.zip",
        target_path="scene_datasets/replica_cad_dataset/rearrange",
    )

    DATA_SOURCES["AI2THOR"] = DataSource(
        source_type="scene",
        url="https://huggingface.co/datasets/haosulab/AI2THOR/resolve/main/ai2thor.zip",
        target_path="scene_datasets/ai2thor",
    )


def initialize_extra_sources():
    DATA_SOURCES["xmate3_robotiq"] = DataSource(
        source_type="robot",
        url="https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/xmate3_robotiq.zip",
        target_path="robots/xmate3_robotiq",
        checksum="ddda102a20eb41e28a0a501702e240e5d7f4084221a44f580e729f08b7c12d1a",
    )
    DATA_SOURCES["ur10e"] = DataSource(
        source_type="robot",
        url="https://github.com/haosulab/ManiSkill-UR10e/archive/refs/tags/v0.1.0.zip",
        target_path="robots/ur10e",
    )
    DATA_SOURCES["anymal_c"] = DataSource(
        source_type="robot",
        url="https://github.com/haosulab/ManiSkill-ANYmalC/archive/refs/tags/v0.1.1.zip",
        target_path="robots/anymal_c",
    )
    DATA_SOURCES["unitree_h1"] = DataSource(
        source_type="robot",
        url="https://github.com/haosulab/ManiSkill-UnitreeH1/archive/refs/tags/v0.1.0.zip",
        target_path="robots/unitree_h1",
    )
    DATA_SOURCES["unitree_go2"] = DataSource(
        source_type="robot",
        url="https://github.com/haosulab/ManiSkill-UnitreeGo2/archive/refs/tags/v0.1.0.zip",
        target_path="robots/unitree_go2",
    )
    DATA_SOURCES["stompy"] = DataSource(
        source_type="robot",
        url="https://github.com/haosulab/ManiSkill-Stompy/archive/refs/tags/v0.1.0.zip",
        target_path="robots/stompy",
    )
    DATA_SOURCES["widowx250s"] = DataSource(
        source_type="robot",
        url="https://github.com/haosulab/ManiSkill-WidowX250S/archive/refs/tags/v0.1.0.zip",
        target_path="robots/widowx",
    )
    DATA_SOURCES["googlerobot"] = DataSource(
        source_type="robot",
        url="https://github.com/haosulab/ManiSkill-GoogleRobot/archive/refs/tags/v0.1.0.zip",
        target_path="robots/googlerobot",
    )

    # ---------------------------------------------------------------------------- #
    # Visual backgrounds
    # ---------------------------------------------------------------------------- #

    # All backgrounds
    # DATA_GROUPS["backgrounds"] = ["minimalistic_modern_bedroom"]

    # TODO add Replica, MatterPort 3D?


def prompt_yes_no(message):
    r"""Prints a message and prompts the user for "y" or "n" returning True or False."""
    # https://github.com/facebookresearch/habitat-sim/blob/main/src_python/habitat_sim/utils/datasets_download.py
    print("\n-------------------------")
    print(message)
    while True:
        answer = input("(y|n): ")
        if answer.lower() == "y":
            return True
        elif answer.lower() == "n":
            return False
        else:
            print("Invalid answer...")


def sha256sum(filename, chunk_size=4096):
    """Computes the SHA256 checksum of a file.

    See also:
        https://www.quickprogrammingtips.com/python/how-to-calculate-sha256-hash-of-a-file-in-python.html
    """
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def download_from_hf_datasets(
    data_source: DataSource,
):
    output_dir = Path(data_source.output_dir)
    output_path = output_dir / data_source.target_path
    snapshot_download(
        repo_id=data_source.hf_repo_id,
        repo_type="dataset",
        local_dir=output_path,
        local_dir_use_symlinks=False,
        resume_download=True,
    )


def download(
    data_source: DataSource,
    verbose=True,
    non_interactive=True,
):
    output_dir = Path(data_source.output_dir)
    # Create output directory
    if not output_dir.exists():
        if non_interactive or prompt_yes_no(f"{output_dir} does not exist. Create?"):
            output_dir.mkdir(parents=True)
        else:
            return
    target_path = data_source.target_path
    if target_path is None:
        target_path = data_source.url.split("/")[-1]
    output_path = output_dir / target_path
    output_dir = osp.dirname(output_path)

    # Clean up existing files
    if output_path.exists():
        if non_interactive or prompt_yes_no(f"{output_path} exists. Remove?"):
            if output_path.is_dir():
                shutil.rmtree(output_path)
            else:
                output_path.unlink()
        else:
            if non_interactive or prompt_yes_no("Continue downloading?"):
                pass
            else:
                print(f"Skip existing: {output_path}")
                return output_path
    output_path.mkdir(parents=True, exist_ok=True)
    if data_source.hf_repo_id is not None:
        download_from_hf_datasets(data_source)
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if data_source.hf_repo_id is not None:
        download_from_hf_datasets(data_source)
        return

    # Download files to temporary location
    try:
        if verbose:
            print(f"Downloading {data_source.url}")
            pbar = tqdm(unit="iB", unit_scale=True, unit_divisor=1024)

        def show_progress(blocknum, bs, size):
            if verbose:
                if blocknum == 0:
                    pbar.total = size
                pbar.update(bs)

        tmp_filename, _ = urllib.request.urlretrieve(
            data_source.url, reporthook=show_progress
        )
        if verbose:
            pbar.close()
    except URLError as err:
        print(f"Failed to download {data_source.url}")
        raise err
    # Verify checksum
    if data_source.checksum is not None and data_source.checksum != sha256sum(
        tmp_filename
    ):
        raise IOError(
            f"Downloaded file's SHA-256 hash does not match record: {data_source.url}"
        )
    base_filename = data_source.filename
    if base_filename is None:
        base_filename = data_source.url.split("/")[-1]
    # Extract or move to output path
    if data_source.url.endswith(".zip"):
        with zipfile.ZipFile(tmp_filename, "r") as zip_ref:
            if verbose:
                for file in tqdm(zip_ref.infolist()):
                    zip_ref.extract(file, output_dir)
            else:
                zip_ref.extractall(output_dir)
            shared_base_dir = None
            for file in zip_ref.filelist:
                base_dir = file.filename.split("/")[0]
                if shared_base_dir is None:
                    shared_base_dir = base_dir
                elif shared_base_dir != base_dir:
                    shared_base_dir = None
            if shared_base_dir is not None and shared_base_dir != osp.basename(
                data_source.target_path
            ):
                os.rename(
                    osp.join(output_dir, shared_base_dir),
                    osp.join(output_dir, osp.basename(data_source.target_path)),
                )
    else:
        shutil.move(tmp_filename, output_path / base_filename)

    # Explicitly delete the temporary file
    if Path(tmp_filename).exists():
        Path(tmp_filename).unlink()

    return output_path


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "uid",
        type=str,
        default="",
        nargs="?",
        help="Asset UID. Use 'all' to download all assets.",
    )
    parser.add_argument(
        "-l",
        "--list",
        type=str,
        help="List all assets available for donwload through this script in a given category. There are 3 categories: 'scene', 'robot', 'task_assets', and 'objects'",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable verbose output.")
    parser.add_argument(
        "-y", "--non-interactive", action="store_true", help="Disable prompts."
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, help="Directory to save assets."
    )
    return parser.parse_args(args)


def main(args):
    global DATA_SOURCES, DATA_GROUPS
    verbose = not args.quiet

    initialize_sources()
    initialize_extra_sources()

    if args.list:
        downloadable_ids = []
        for k, v in DATA_SOURCES.items():
            if v.source_type == args.list:
                downloadable_ids.append(k)
        print(f"For category {args.list} the following asset UIDs are available")
        print(downloadable_ids)
        exit()
    if args.uid == "":
        print("Available asset (group) uids:")
        print(list(DATA_GROUPS.keys()))
        return
    if args.uid == "all":
        if verbose:
            print("All assets will be downloaded. This may take a while.")
        uids = list(DATA_SOURCES.keys())
        show_progress = True
    elif args.uid in DATA_GROUPS:
        uids = DATA_GROUPS[args.uid]
        show_progress = True
    elif args.uid in DATA_SOURCES:
        uids = [args.uid]
        show_progress = False
    else:
        raise KeyError("{} not found.".format(args.uid))

    for i, uid in enumerate(uids):
        if show_progress and verbose:
            print("Downloading assets for {}: {}/{}".format(args.uid, i + 1, len(uids)))

        kwargs = dict()
        kwargs["verbose"] = verbose
        kwargs["non_interactive"] = args.non_interactive
        if args.output_dir is not None:
            kwargs["output_dir"] = args.output_dir
        output_path = download(DATA_SOURCES[uid], **kwargs)

        if output_path is not None and verbose:
            print("=" * 80)
            print(f"Asset ({uid}) is successfully downloaded to {output_path}.")
            print("=" * 80)


if __name__ == "__main__":
    args = parse_args()
    main(args)
