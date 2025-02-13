import argparse
import hashlib
import os
import os.path as osp
import shutil
import urllib.request
import zipfile
from pathlib import Path
from urllib.error import URLError

from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

import mani_skill.envs  # import all environments to register them which auto registers data groups to allow asset download by environment ID.
from mani_skill.utils import assets
from mani_skill.utils.assets.data import DATA_GROUPS, DATA_SOURCES


def prompt_yes_no(message):
    skip_prompt = os.getenv("MS_SKIP_ASSET_DOWNLOAD_PROMPT")
    if skip_prompt == "1":
        return True
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
    data_source: assets.DataSource,
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
    data_source: assets.DataSource,
    verbose=True,
    non_interactive=True,
):
    """download a given data source"""
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
    verbose = not args.quiet

    if args.list:
        downloadable_ids = []
        for k, v in assets.DATA_SOURCES.items():
            if v.source_type == args.list:
                downloadable_ids.append(k)
        print(f"For category {args.list} the following asset UIDs are available")
        print(downloadable_ids)
        exit()
    if args.uid == "":
        print("Available asset (group) uids:")
        print(list(assets.DATA_GROUPS.keys()))
        return
    if args.uid == "all":
        if verbose:
            print("All assets will be downloaded. This may take a while.")
        uids = list(assets.DATA_SOURCES.keys())
        show_progress = True
    elif args.uid in assets.DATA_GROUPS:
        uids = assets.expand_data_group_into_individual_data_source_ids(args.uid)
        show_progress = True
    elif args.uid in assets.DATA_SOURCES:
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
        output_path = download(assets.DATA_SOURCES[uid], **kwargs)

        if output_path is not None and verbose:
            print("=" * 80)
            print(f"Asset ({uid}) is successfully downloaded to {output_path}.")
            print("=" * 80)


if __name__ == "__main__":
    args = parse_args()
    main(args)
