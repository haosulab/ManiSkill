"""Utilities to download assets.

See also:
    https://github.com/facebookresearch/habitat-sim/blob/main/src_python/habitat_sim/utils/datasets_download.py
    https://github.com/StanfordVL/iGibson/blob/master/igibson/utils/assets_utils.py
"""

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path
from urllib.error import URLError

from tqdm.auto import tqdm

from mani_skill2 import ASSET_DIR, PACKAGE_ASSET_DIR
from mani_skill2.utils.io_utils import load_json

DATA_SOURCES = {}
DATA_GROUPS = {}


def initialize_sources():
    DATA_SOURCES["ycb"] = dict(
        url="https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/mani_skill2_ycb_v2.zip",
        target_path="mani_skill2_ycb",
        checksum="b83afbe9c38a780f5625f2c97afad712db49fcb533d4814ac0c827b0514a504b",
    )
    DATA_GROUPS["PickSingleYCB-v0"] = ["ycb"]

    DATA_SOURCES["egad"] = dict(
        url="https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/mani_skill2_egad_v1.zip",
        target_path="mani_skill2_egad",
        checksum="4b5d841256e0151c2a615a98d4e92afa12ad6d795e2565b364586e3940d3aa36",
    )
    DATA_GROUPS["PickSingleEGAD-v0"] = ["egad"]

    DATA_SOURCES["pick_clutter_ycb"] = dict(
        url="https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/pick_clutter/ycb_train_5k.json.gz",
        output_dir=ASSET_DIR / "pick_clutter",
        checksum="70ec176c7036f326ea7813b77f8c03bea9db5960198498957a49b2895a9ec338",
    )
    DATA_GROUPS["PickClutterYCB-v0"] = ["ycb", "pick_clutter_ycb"]

    DATA_SOURCES["assembling_kits"] = dict(
        url="https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/assembling_kits_v1.zip",
        target_path="assembling_kits",
        checksum="e3371f17a07a012edaa3a0b3604fb1577f3fb921876c3d5ed59733dd75a6b4a0",
    )
    DATA_GROUPS["AssemblingKits-v0"] = ["assembling_kits"]

    DATA_SOURCES["panda_avoid_obstacles"] = dict(
        url="https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/avoid_obstacles/panda_train_2k.json.gz",
        output_dir=ASSET_DIR / "avoid_obstacles",
        checksum="44dae9a0804172515c290c1f49a1e7e72d76e40201a2c5c7d4a3ccd43b4d5be4",
    )
    DATA_GROUPS["PandaAvoidObstacles-v0"] = ["panda_avoid_obstacles"]

    DATA_SOURCES["pinch"] = dict(
        url="https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/pinch.zip",
        target_path="pinch",
        checksum="3281d2d777fad42e6d37371b2d3ee16fb1c39984907176718ca2e4f447326fe7",
    )
    DATA_GROUPS["Pinch-v0"] = ["pinch"]

    DATA_SOURCES["write"] = dict(
        url="https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/write.zip",
        target_path="write",
        checksum="c5b49e581bfed9cfb2107a607faf52795f840e93f5a7ad389290314513b4b634",
    )
    DATA_GROUPS["Write-v0"] = ["write"]

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
            DATA_SOURCES[uid] = dict(
                url=f"https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/partnet_mobility/dataset/{model_id}.zip",
                output_dir=ASSET_DIR / "partnet_mobility" / "dataset",
            )
            category_uids[category].append(uid)

    DATA_GROUPS["partnet_mobility_cabinet"] = set(
        category_uids["cabinet_drawer"] + category_uids["cabinet_door"]
    )
    DATA_GROUPS["partnet_mobility_chair"] = category_uids["chair"]
    DATA_GROUPS["partnet_mobility_bucket"] = category_uids["bucket"]
    DATA_GROUPS["partnet_mobility_faucet"] = category_uids["faucet"]

    DATA_GROUPS["OpenCabinetDrawer-v1"] = category_uids["cabinet_drawer"]
    DATA_GROUPS["OpenCabinetDoor-v1"] = category_uids["cabinet_door"]
    DATA_GROUPS["PushChair-v1"] = category_uids["chair"]
    DATA_GROUPS["MoveBucket-v1"] = category_uids["bucket"]
    DATA_GROUPS["TurnFaucet-v0"] = category_uids["faucet"]


def initialize_extra_sources():
    DATA_SOURCES["xmate3_robotiq"] = dict(
        url="https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/xmate3_robotiq.zip",
        target_path="xmate3_robotiq",
        checksum="ddda102a20eb41e28a0a501702e240e5d7f4084221a44f580e729f08b7c12d1a",
    )

    # ---------------------------------------------------------------------------- #
    # Visual backgrounds
    # ---------------------------------------------------------------------------- #
    DATA_SOURCES["minimalistic_modern_bedroom"] = dict(
        url="https://storage1.ucsd.edu/datasets/ManiSkill-background/minimalistic_modern_bedroom.glb",
        output_dir=ASSET_DIR / "background",
        checksum="9d9ea14c8cdfab1ebafe3e8ff5a071b77b361c1abd24e9c59b0f08e1d1a3421a",
    )
    # Alias
    DATA_GROUPS["minimal_bedroom"] = ["minimalistic_modern_bedroom"]

    # All backgrounds
    DATA_GROUPS["backgrounds"] = ["minimalistic_modern_bedroom"]


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


def download(
    url,
    output_dir=ASSET_DIR,
    target_path: str = None,
    checksum=None,
    verbose=True,
    non_interactive=True,
):
    output_dir = Path(output_dir)

    # Create output directory
    if not output_dir.exists():
        if non_interactive or prompt_yes_no(f"{output_dir} does not exist. Create?"):
            output_dir.mkdir(parents=True)
        else:
            return

    if target_path is None:
        target_path = url.split("/")[-1]
    output_path = output_dir / target_path

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

    # Download files to temporary location
    try:
        if verbose:
            print(f"Downloading {url}")
            pbar = tqdm(unit="iB", unit_scale=True, unit_divisor=1024)

        def show_progress(blocknum, bs, size):
            if verbose:
                if blocknum == 0:
                    pbar.total = size
                pbar.update(bs)

        filename, _ = urllib.request.urlretrieve(url, reporthook=show_progress)
        if verbose:
            pbar.close()
    except URLError as err:
        print(f"Failed to download {url}")
        raise err

    # Verify checksum
    if checksum is not None and checksum != sha256sum(filename):
        raise IOError(f"Downloaded file's SHA-256 hash does not match record: {url}")

    # Extract or move to output path
    if url.endswith(".zip"):
        with zipfile.ZipFile(filename, "r") as zip_ref:
            if verbose:
                for file in tqdm(zip_ref.infolist()):
                    zip_ref.extract(file, output_dir)
            else:
                zip_ref.extractall(output_dir)
    else:
        shutil.move(filename, output_path)

    # Explicitly delete the temporary file
    if Path(filename).exists():
        Path(filename).unlink()

    return output_path


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "uid",
        type=str,
        help="Asset UID. Use 'all' to download all assets.",
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
    verbose = not args.quiet

    initialize_sources()
    initialize_extra_sources()

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

        kwargs = DATA_SOURCES[uid].copy()
        kwargs["verbose"] = verbose
        kwargs["non_interactive"] = args.non_interactive
        if args.output_dir is not None:
            kwargs["output_dir"] = args.output_dir
        output_path = download(**kwargs)

        if output_path is not None and verbose:
            print("=" * 80)
            print(f"Asset ({uid}) is successfully downloaded to {output_path}.")
            print("=" * 80)


if __name__ == "__main__":
    args = parse_args()
    main(args)
