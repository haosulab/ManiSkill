import argparse
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from tqdm.auto import tqdm

from mani_skill2 import ASSET_DIR
from mani_skill2.utils.io_utils import load_json

DATA_SOURCES = {}
DATA_SOURCES["ycb"] = dict(
    url="https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/mani_skill2_ycb_v1.zip",
    checksum="adc9c0931cdb8f8d304f1bcff2446a084682ec8a79c01f2081b2d2cfaedc629c",
    output_dir=ASSET_DIR,
)
DATA_SOURCES["egad"] = dict(
    url="https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/mani_skill2_egad.zip",
    checksum="a06f335148a1aa420625f89ef94674dbd59932c6d86beb20784c095c7f915970",
    output_dir=ASSET_DIR,
)
DATA_SOURCES["faucet"] = dict(
    url="https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/partnet_mobility/dataset/{}.zip",
    output_dir=ASSET_DIR / "partnet_mobility" / "dataset",
    model_db=load_json(ASSET_DIR / "partnet_mobility/meta/info_faucet_train.json"),
)
DATA_SOURCES["avoid_obstacles"] = dict(
    url="https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/avoid_obstacles/panda_train_2k.json.gz",
    output_dir=ASSET_DIR / "avoid_obstacles",
)
DATA_SOURCES["assembling_kits"] = dict(
    url="https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/assembling_kits.zip",
    checksum="be93332248187975d4dc2ca172ef00b1774c51f71f51ae10502218b17f5206a6",
    output_dir=ASSET_DIR,
)
DATA_SOURCES["pinch"] = dict(
    url="https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/pinch.zip",
    checksum="3281d2d777fad42e6d37371b2d3ee16fb1c39984907176718ca2e4f447326fe7",
    output_dir=ASSET_DIR,
)
DATA_SOURCES["write"] = dict(
    url="https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/write.zip",
    checksum="c5b49e581bfed9cfb2107a607faf52795f840e93f5a7ad389290314513b4b634",
    output_dir=ASSET_DIR,
)

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


def download(
    url,
    output_dir,
    checksum=None,
    ignore_exist=False,
    chunk_size=1024 * 32,
    verbose=True,
    model_db=None,
):
    if model_db is not None:
        if verbose:
            print(f"Downloading {len(model_db)} models")
            for model_id in tqdm(model_db):
                download(url.format(model_id), output_dir, verbose=False)
        else:
            for model_id in model_db:
                download(url.format(model_id), output_dir, verbose=False)
        return

    r = urllib.request.urlopen(url)
    if r.status != 200:
        raise ConnectionError(r.status)

    output_dir = Path(output_dir)
    if output_dir.exists() and ignore_exist:
        to_remove = prompt_yes_no("{} exists. Remove?") if verbose else True
        if to_remove:
            shutil.rmtree(output_dir)

    hasher = hashlib.sha256()
    if verbose:
        print(f"Downloading {url}...")

    fp = (
        tempfile.TemporaryFile()
        if url.endswith("zip")
        else open(output_dir / url.split("/")[-1], "wb")
    )

    pbar = (
        tqdm(total=r.length, unit="iB", unit_scale=True, unit_divisor=1024)
        if verbose
        else None
    )

    for chunk in iter(lambda: r.read(chunk_size), b""):
        # filter out keep-alive new chunks
        if not chunk:
            continue

        fp.write(chunk)
        hasher.update(chunk)

        if pbar is not None:
            pbar.update(len(chunk))

    if pbar is not None:
        pbar.close()

    if checksum is not None and hasher.hexdigest() != checksum:
        raise IOError("Downloaded file's SHA-256 hash does not match record")

    if url.endswith("zip"):
        z = zipfile.ZipFile(fp)
        if verbose:
            print(f"Extracting {url} to {output_dir}...")
            for file in tqdm(z.infolist()):
                z.extract(file, output_dir)
        else:
            z.extractall(output_dir)

    fp.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uid", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.uid == "all":
        print("Downloading all assets")
        uids = list(DATA_SOURCES.keys())
    else:
        uids = [args.uid]

    for uid in uids:
        kwargs = DATA_SOURCES[uid]
        download(**kwargs)
        print("=" * 50)
        print(
            "Dataset ({}) successfully downloaded to {}.".format(
                args.uid, kwargs["output_dir"]
            )
        )
        print("=" * 50)


if __name__ == "__main__":
    main()
