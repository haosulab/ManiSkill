"""Upload the ManiSkill PickCube LeRobot dataset to the Hugging Face Hub.

Requires an HF token with write access (`hf auth login`).
Defaults to a PRIVATE repo; pass --public to publish publicly.
"""

import argparse

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = "/home/kelin/dataset/maniskill/pickcube/lerobot"
SRC_REPO_ID = "maniskill/pickcube"
DST_REPO_ID = "KelinLiIC/maniskill-test"

parser = argparse.ArgumentParser()
parser.add_argument("--public", action="store_true", help="create a public repo instead of private")
args = parser.parse_args()

private = not args.public

ds = LeRobotDataset(SRC_REPO_ID, root=ROOT, video_backend="pyav")

# push_to_hub() uses self.repo_id for create_repo/upload_folder/card/tag.
ds.repo_id = DST_REPO_ID
ds.meta.repo_id = DST_REPO_ID

print(f"Uploading {ds.num_episodes} episodes / {ds.num_frames} frames")
print(f"  -> {DST_REPO_ID} (private={private})")

ds.push_to_hub(
    private=private,
    tags=["maniskill", "robotics", "panda", "simulation", "pick-cube"],
    push_videos=True,
)

print(f"\nDone: https://huggingface.co/datasets/{DST_REPO_ID}")
