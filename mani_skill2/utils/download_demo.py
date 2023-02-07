import argparse
import os.path as osp

import gdown

VALID_ENV_TYPES = ["rigid_body", "soft_body"]

DATASET_SOURCES = {}

# Rigid body envs
DATASET_SOURCES["LiftCube-v0"] = dict(
    env_type="rigid_body",
    s1_url="",
    gd_url="https://drive.google.com/drive/folders/1TZ2rWOWHELT66JA8S-Z3aWKC7pM-Kw8l?usp=share_link",
)
DATASET_SOURCES["PickClutterYCB-v0"] = dict(
    env_type="rigid_body",
    s1_url="",
    gd_url="https://drive.google.com/drive/folders/1sokvFs5ptWMIbnqo1VEF1pB9PtSv4dy_?usp=share_link",
)
DATASET_SOURCES["AssemblingKits-v0"] = dict(
    env_type="rigid_body",
    s1_url="",
    gd_url="https://drive.google.com/drive/folders/17TrTNKEvhA9cJNx-EQ2QY64Zrw-Tcykk?usp=share_link",
)
DATASET_SOURCES["TurnFaucet-v0"] = dict(
    env_type="rigid_body",
    s1_url="",
    gd_url="https://drive.google.com/drive/folders/1YIxqaccA48gxfRjVQLpjomgH8T48P9wf?usp=share_link",
)
DATASET_SOURCES["PandaAvoidObstacles-v0"] = dict(
    env_type="rigid_body",
    s1_url="",
    gd_url="https://drive.google.com/drive/folders/1RooBQfswwU4cyqOBd2LRiJ7dyVRx0Y9d?usp=share_link",
)
DATASET_SOURCES["PickSingleEGAD-v0"] = dict(
    env_type="rigid_body",
    s1_url="",
    gd_url="https://drive.google.com/drive/folders/1qtU9yRZC1ApEUZ-5K1NgaUbHn0yBJauC?usp=share_link",
)
DATASET_SOURCES["PickSingleYCB-v0"] = dict(
    env_type="rigid_body",
    s1_url="",
    gd_url="https://drive.google.com/drive/folders/1HQUws_Aoiw44viXopgJtxS5675kOqQfc?usp=share_link",
)
DATASET_SOURCES["PlugCharger-v0"] = dict(
    env_type="rigid_body",
    s1_url="",
    gd_url="https://drive.google.com/drive/folders/1E0ge305MIW2FindLBImeNBEzkv9uEuaG?usp=share_link",
)
DATASET_SOURCES["PegInsertionSide-v0"] = dict(
    env_type="rigid_body",
    s1_url="",
    gd_url="https://drive.google.com/drive/folders/1QCrBPxtODF-v3k3tQEckoDnSz0GzWoji?usp=share_link",
)
DATASET_SOURCES["StackCube-v0"] = dict(
    env_type="rigid_body",
    s1_url="",
    gd_url="https://drive.google.com/drive/folders/1XSSsI58rpLYxyexbNFf7LzhJQXAY_fVO?usp=share_link",
)
DATASET_SOURCES["PickCube-v0"] = dict(
    env_type="rigid_body",
    s1_url="",
    gd_url="https://drive.google.com/drive/folders/1WgYpQjqnZBbyXqlqtQfoNlCKAVPdeRIx?usp=share_link",
)
# Soft body envs
DATASET_SOURCES["Write-v0"] = dict(
    env_type="soft_body",
    s1_url="",
    gd_url="https://drive.google.com/drive/folders/1ziY-cL_ofq52zYapdQIJNJWnaZZPGVO9?usp=share_link",
)
DATASET_SOURCES["Pinch-v0"] = dict(
    env_type="soft_body",
    s1_url="",
    gd_url="https://drive.google.com/drive/folders/1F6Yx1kqZ8mg2H0ShqNWnwsTEigcnAEol?usp=share_link",
)
DATASET_SOURCES["Hang-v0"] = dict(
    env_type="soft_body",
    s1_url="",
    gd_url="https://drive.google.com/drive/folders/1zpU__pG-N7SEhIH4YD87MeeGRDfoGq5_?usp=share_link",
)
DATASET_SOURCES["Pour-v0"] = dict(
    env_type="soft_body",
    s1_url="",
    gd_url="https://drive.google.com/drive/folders/1jlZ3K6nOXgqjXeKVRYNKdXIbTRau_1Ge?usp=share_link",
)
DATASET_SOURCES["Excavate-v0"] = dict(
    env_type="soft_body",
    s1_url="",
    gd_url="https://drive.google.com/drive/folders/1G3eycgS1ckJlG5R5Z6wClCAPHuSg94ei?usp=share_link",
)
DATASET_SOURCES["Fill-v0"] = dict(
    env_type="soft_body",
    s1_url="",
    gd_url="https://drive.google.com/drive/folders/1Fz57NrZP-tanQM5sSN8iuImIqZboR3X2?usp=share_link",
)


def download_demo(env_id: str, output_dir: str):
    """
    download a demonstration dataset for a specific environment
    """
    if env_id not in DATASET_SOURCES:
        raise ValueError(f"{env_id} is not a valid environment id with demonstrations")
    dataset_metadata = DATASET_SOURCES[env_id]
    output_dir = osp.join(output_dir, dataset_metadata["env_type"], env_id)

    # we first try the google drive link
    print(f"Downloading from Google Drive link {dataset_metadata['gd_url']} ...")

    result = gdown.download_folder(dataset_metadata["gd_url"], output=output_dir)
    if result == False:
        print("Google drive link failed")
    # TODO add mirror links

    if result == False:
        print(
            "All links failed, please report this issue to the maintainers by posting an issue to https://github.com/haosulab/ManiSkill2/issues"
        )


def download_all_demos(output_dir: str):
    """
    calls download_demo for all environments with demonstrations available
    """
    for env_id in DATASET_SOURCES.keys():
        download_demo(env_id, output_dir)


def download_demos_by_type(env_type: str, output_dir: str):
    """
    calls download_demo for all environments with demonstrations available with matching environment type
    """
    download_count = 0
    for env_id in DATASET_SOURCES.keys():
        if DATASET_SOURCES[env_id]["env_type"] == env_type:
            download_demo(env_id, output_dir)
            download_count += 1
    if download_count == 0:
        print(f"Env type {env_type} has no available demonstrations")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "env",
        type=str,
        help="Specify either 'all', an env type (e.g. rigid_body, soft_body), or a specific env id (e.g. PickCube-v0)",
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
    if args.env in VALID_ENV_TYPES:
        download_demos_by_type(args.env, output_dir=args.output_dir)
    elif args.env == "all":
        print("Downloading all demonstrations for all environments")
        download_all_demos(output_dir=args.output_dir)
    else:
        download_demo(args.env, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
