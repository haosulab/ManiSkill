"""GitPython based utility to download robot assets from a git repository.

Many robots used in ManiSkill are distributed via standalone git repositories
(e.g. community robot description repos). This module provides a simple way to
shallow clone such a repository into the local asset directory (by default
``~/.maniskill/data/robots``), optionally pinned to a specific commit so that
downloads are reproducible.

Requires GitPython which is already a dependency of ManiSkill.
"""

import argparse
import os
import os.path as osp
from pathlib import Path
from typing import Optional

import git

from mani_skill import ASSET_DIR

DEFAULT_ROBOT_ASSET_DIR = ASSET_DIR / "robots"
"""Default directory robot repos are cloned into."""


def _shallow_fetch_commit(repo: git.Repo, commit: str):
    """Try to fetch a single commit shallowly, falling back to a full fetch."""
    try:
        repo.git.fetch("origin", commit, "--depth=1")
    except git.GitCommandError:
        # Some git servers refuse fetching arbitrary SHA-1s
        # (uploadpack.allowReachableSHA1InWant / allowAnySHA1InWant are off),
        # so fall back to fetching all history.
        repo.git.fetch("origin")


def _clone(repo_url: str, repo_path: Path) -> git.Repo:
    """Shallow clone ``repo_url`` into ``repo_path``.

    ``--no-local`` is needed for local paths so that git does not shortcut to a
    filesystem copy, which would ignore ``--depth``.
    """
    kwargs = dict(depth=1)
    if osp.exists(repo_url) or repo_url.startswith("file://"):
        kwargs["multi_options"] = ["--no-local"]
    return git.Repo.clone_from(repo_url, repo_path, **kwargs)


def download_robot_repo(
    repo_url: str,
    commit: Optional[str] = None,
    target_dir: Optional[os.PathLike] = None,
    repo_name: Optional[str] = None,
    force: bool = False,
) -> Path:
    """Clone a robot asset repository from ``repo_url`` and return the local path.

    Args:
        repo_url: URL of the git repository (https, ssh, or a local path / file:// URL).
        commit: optional commit SHA to pin the download to. If None, the default
            branch HEAD is used.
        target_dir: directory to clone into. Defaults to ``ASSET_DIR / robots``.
        repo_name: name of the directory the repo is cloned to. Defaults to the
            repository name in ``repo_url``.
        force: if True, re-clone from scratch even if the repo already exists
            locally at the requested commit.

    The download is idempotent: if the repository already exists locally and is
    already at the requested commit, nothing is re-downloaded. Otherwise the
    new commits are fetched and the requested commit is checked out.
    """
    if repo_name is None:
        repo_name = osp.splitext(osp.basename(repo_url.rstrip("/")))[0]
    if target_dir is None:
        target_dir = DEFAULT_ROBOT_ASSET_DIR
    target_dir = Path(target_dir)
    repo_path = target_dir / repo_name

    if repo_path.exists():
        try:
            repo = git.Repo(repo_path)
        except git.InvalidGitRepositoryError:
            if not force:
                raise FileExistsError(
                    f"{repo_path} exists but is not a git repository; "
                    "remove it or pass force=True to replace it."
                )
            import shutil

            shutil.rmtree(repo_path)
            repo = _clone(repo_url, repo_path)
        if repo is not None:
            if commit is not None and str(repo.head.commit) == commit:
                return repo_path
            if commit is None:
                # Already cloned, nothing more requested.
                return repo_path
            _shallow_fetch_commit(repo, commit)
            repo.git.checkout(commit)
            return repo_path
    else:
        target_dir.mkdir(parents=True, exist_ok=True)
        # Shallow clone the default branch; we pin to a commit afterwards if
        # one is requested.
        repo = _clone(repo_url, repo_path)

    if commit is not None and str(repo.head.commit) != commit:
        _shallow_fetch_commit(repo, commit)
        repo.git.checkout(commit)
    return repo_path


def main():
    parser = argparse.ArgumentParser(
        description="Download robot assets from a git repository."
    )
    parser.add_argument("repo_url", help="URL of the git repository to download")
    parser.add_argument(
        "--commit", default=None, help="optional commit SHA to pin the download to"
    )
    parser.add_argument(
        "--target-dir",
        default=None,
        help=f"directory to clone into (default: {DEFAULT_ROBOT_ASSET_DIR})",
    )
    parser.add_argument(
        "--repo-name",
        default=None,
        help="name of the directory the repo is cloned to (default: repository name)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-clone from scratch even if the repo already exists locally",
    )
    args = parser.parse_args()
    repo_path = download_robot_repo(
        repo_url=args.repo_url,
        commit=args.commit,
        target_dir=args.target_dir,
        repo_name=args.repo_name,
        force=args.force,
    )
    print(f"Robot assets downloaded to {repo_path}")


if __name__ == "__main__":
    main()
