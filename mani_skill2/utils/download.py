import argparse
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
    url="https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/mani_skill2_ycb.zip",
    output_dir=ASSET_DIR,
)
DATA_SOURCES["egad"] = dict(
    url="https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/mani_skill2_egad.zip",
    output_dir=ASSET_DIR,
)
DATA_SOURCES["faucet"] = dict(
    url="https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/partnet_mobility/dataset/{}.zip",
    output_dir=ASSET_DIR / "partnet_mobility" / "dataset",
    model_db=load_json(ASSET_DIR / "partnet_mobility/meta/info_faucet_train.json"),
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

    if verbose:
        print(f"Downloading {url}...")
    with tempfile.TemporaryFile() as fp:
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
            if pbar is not None:
                pbar.update(len(chunk))
        if pbar is not None:
            pbar.close()

        z = zipfile.ZipFile(fp)
        if verbose:
            print(f"Extracting {url} to {output_dir}...")
            for file in tqdm(z.infolist()):
                z.extract(file, output_dir)
        else:
            z.extractall(output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uid", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    kwargs = DATA_SOURCES[args.uid]
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
