"""
Run python -m mani_skill .utils.scene_builder.ai2thor.download <HF_TOKEN>
"""

import os.path as osp
import sys

from huggingface_hub import login, snapshot_download

from mani_skill import ASSET_DIR

if __name__ == "__main__":
    login(sys.argv[1])
    snapshot_download(
        repo_id="hssd/ai2thor-hab",
        repo_type="dataset",
        local_dir=osp.join(ASSET_DIR, "scene_datasets/ai2thor"),
        local_dir_use_symlinks=False,
        allow_patterns=["**/*"],
        resume_download=True,
    )
