from pathlib import Path

from .utils.logging_utils import logger

ROOT_DIR = Path(__file__).parent.resolve()
ASSET_DIR = ROOT_DIR / "assets"
AGENT_CONFIG_DIR = ASSET_DIR / "config_files/agents"
DESCRIPTION_DIR = ASSET_DIR / "descriptions"


def get_commit_info(show_modified_files=False, show_untracked_files=False):
    """Get git commit information."""
    # isort: off
    import git

    try:
        repo = git.Repo(ROOT_DIR.parent)
    except git.InvalidGitRepositoryError as err:
        logger.warning("mani_skill2 is not installed with git.")
        return None
    else:
        commit_info = {}
        commit_info["commit_id"] = str(repo.head.commit)
        commit_info["branch"] = (
            None if repo.head.is_detached else repo.active_branch.name
        )

        if show_modified_files:
            # https://stackoverflow.com/questions/33733453/get-changed-files-using-gitpython
            modified_files = [item.a_path for item in repo.index.diff(None)]
            commit_info["modified"] = modified_files

        if show_untracked_files:
            untracked_files = repo.untracked_files
            commit_info["untracked"] = modified_files

        # https://github.com/gitpython-developers/GitPython/issues/718#issuecomment-360267779
        repo.__del__()
        return commit_info
