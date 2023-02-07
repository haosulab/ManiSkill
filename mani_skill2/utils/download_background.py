#!/usr/bin/env python3

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


DATA_SOURCES["minimalistic_modern_bedroom"] = dict(
    url="https://storage1.ucsd.edu/datasets/ManiSkill-background/minimalistic_modern_bedroom.glb",
    target_path="minimalistic_modern_bedroom.glb",
    checksum="9d9ea14c8cdfab1ebafe3e8ff5a071b77b361c1abd24e9c59b0f08e1d1a3421a",
)


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
            if blocknum == 0:
                pbar.total = size
            pbar.update(bs)

        filename, _ = urllib.request.urlretrieve(url, reporthook=show_progress)
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


if __name__ == "__main__":
    for _, entry in DATA_SOURCES.items():
        download(
            entry["url"],
            output_dir=ASSET_DIR / "background",
            target_path=entry["target_path"],
            checksum=entry["checksum"],
        )
