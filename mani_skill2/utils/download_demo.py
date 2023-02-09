import argparse
import os
import os.path as osp
import zipfile

import gdown

DATASET_SOURCES = {}

# Rigid body envs
DATASET_SOURCES["LiftCube-v0"] = dict(
    env_type="rigid_body",
    gd_url="https://drive.google.com/drive/folders/1TZ2rWOWHELT66JA8S-Z3aWKC7pM-Kw8l?usp=share_link",
)
DATASET_SOURCES["PickClutterYCB-v0"] = dict(
    env_type="rigid_body",
    gd_url="https://drive.google.com/drive/folders/1sokvFs5ptWMIbnqo1VEF1pB9PtSv4dy_?usp=share_link",
)
DATASET_SOURCES["AssemblingKits-v0"] = dict(
    env_type="rigid_body",
    gd_url="https://drive.google.com/drive/folders/17TrTNKEvhA9cJNx-EQ2QY64Zrw-Tcykk?usp=share_link",
)
DATASET_SOURCES["TurnFaucet-v0"] = dict(
    env_type="rigid_body",
    gd_url="https://drive.google.com/uc?id=1KCSaYO_2HtQCCDBDw7twgGcMsD48wRVk",
)
DATASET_SOURCES["PandaAvoidObstacles-v0"] = dict(
    env_type="rigid_body",
    gd_url="https://drive.google.com/drive/folders/1RooBQfswwU4cyqOBd2LRiJ7dyVRx0Y9d?usp=share_link",
)
DATASET_SOURCES["PickSingleEGAD-v0"] = dict(
    env_type="rigid_body",
    gd_url="https://drive.google.com/drive/folders/1qtU9yRZC1ApEUZ-5K1NgaUbHn0yBJauC?usp=share_link",
)
DATASET_SOURCES["PickSingleYCB-v0"] = dict(
    env_type="rigid_body",
    gd_url="https://drive.google.com/uc?id=1fWFhoNC3AnhiI9ZYECryCx77WLDEv3ju",
)
DATASET_SOURCES["PlugCharger-v0"] = dict(
    env_type="rigid_body",
    gd_url="https://drive.google.com/drive/folders/1E0ge305MIW2FindLBImeNBEzkv9uEuaG?usp=share_link",
)
DATASET_SOURCES["PegInsertionSide-v0"] = dict(
    env_type="rigid_body",
    gd_url="https://drive.google.com/drive/folders/1QCrBPxtODF-v3k3tQEckoDnSz0GzWoji?usp=share_link",
)
DATASET_SOURCES["StackCube-v0"] = dict(
    env_type="rigid_body",
    gd_url="https://drive.google.com/drive/folders/1XSSsI58rpLYxyexbNFf7LzhJQXAY_fVO?usp=share_link",
)
DATASET_SOURCES["PickCube-v0"] = dict(
    env_type="rigid_body",
    gd_url="https://drive.google.com/drive/folders/1WgYpQjqnZBbyXqlqtQfoNlCKAVPdeRIx?usp=share_link",
)
DATASET_SOURCES["PushChair-v1"] = dict(
    env_type="rigid_body",
    gd_url="https://drive.google.com/uc?id=1Ppf0R8KlxTcNmENB4BFEAdUgBSZW43sL",
)
DATASET_SOURCES["OpenCabinetDrawer-v1"] = dict(
    env_type="rigid_body",
    gd_url="https://drive.google.com/uc?id=1MSOGvO2EEEakRM-g-dnSaQqOly3eC6Wd",
)
DATASET_SOURCES["OpenCabinetDoor-v1"] = dict(
    env_type="rigid_body",
    gd_url="https://drive.google.com/uc?id=17lPXYhsGtk8l6RSBrusIBagoOGeJQUoC",
)
DATASET_SOURCES["MoveBucket-v1"] = dict(
    env_type="rigid_body",
    gd_url="https://drive.google.com/uc?id=1yu6SeU4Bp-mF0WVOtiz9qTOa-9ABEPsv",
)
# Soft body envs
DATASET_SOURCES["Write-v0"] = dict(
    env_type="soft_body",
    gd_url="https://drive.google.com/drive/folders/1ziY-cL_ofq52zYapdQIJNJWnaZZPGVO9?usp=share_link",
)
DATASET_SOURCES["Pinch-v0"] = dict(
    env_type="soft_body",
    gd_url="https://drive.google.com/drive/folders/1F6Yx1kqZ8mg2H0ShqNWnwsTEigcnAEol?usp=share_link",
)
DATASET_SOURCES["Hang-v0"] = dict(
    env_type="soft_body",
    gd_url="https://drive.google.com/drive/folders/1zpU__pG-N7SEhIH4YD87MeeGRDfoGq5_?usp=share_link",
)
DATASET_SOURCES["Pour-v0"] = dict(
    env_type="soft_body",
    gd_url="https://drive.google.com/drive/folders/1jlZ3K6nOXgqjXeKVRYNKdXIbTRau_1Ge?usp=share_link",
)
DATASET_SOURCES["Excavate-v0"] = dict(
    env_type="soft_body",
    gd_url="https://drive.google.com/drive/folders/1G3eycgS1ckJlG5R5Z6wClCAPHuSg94ei?usp=share_link",
)
DATASET_SOURCES["Fill-v0"] = dict(
    env_type="soft_body",
    gd_url="https://drive.google.com/drive/folders/1Fz57NrZP-tanQM5sSN8iuImIqZboR3X2?usp=share_link",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "uid",
        type=str,
        help="An environment id (e.g. PickCube-v0), a type of environments (rigid_body/soft_body), or 'all' for all available demonstrations.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="demos",
        help="The directory to save demonstrations to. The files will then be saved to <output_dir>/<env_type>/<env_id>",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.uid == "":
        print("Available uids:")
        print(list(DATASET_SOURCES.keys()))
        return

    if args.uid == "all":
        print("All demonstrations will be downloaded. This may take a while.")
        uids = list(DATASET_SOURCES.keys())
        show_progress = True
    elif args.uid in ["rigid_body", "soft_body"]:
        uids = []
        for k, v in DATASET_SOURCES.items():
            if v["env_type"] == args.uid:
                uids.append(k)
        show_progress = True
    elif args.uid in DATASET_SOURCES:
        uids = [args.uid]
        show_progress = False
    else:
        raise KeyError("{} not found.".format(args.uid))

    for i, uid in enumerate(uids):
        if show_progress:
            print("Downloading demonstrations: {}/{}".format(i + 1, len(uids)))

        meta = DATASET_SOURCES[uid]

        url = meta["gd_url"]
        is_folder = "folder" in url
        if is_folder:
            output_path = osp.join(args.output_dir, meta["env_type"], uid)
            filenames = gdown.download_folder(url, output=output_path)
            is_failed = filenames is None
        else:
            output_path = osp.join(args.output_dir, meta["env_type"], uid + ".zip")
            os.makedirs(osp.dirname(output_path), exist_ok=True)
            filename = gdown.download(url, output=output_path)
            is_failed = filename is None

        if is_failed:
            print(f"Google drive link failed: {url}")
        elif not is_folder:
            with zipfile.ZipFile(filename, "r") as zip_ref:
                zip_ref.extractall(osp.dirname(filename))
            print("Unzip file: {}".format(filename))
            os.remove(filename)

        if is_failed:
            print("Failed to download demonstrations for {}".format(uid))


if __name__ == "__main__":
    main()
