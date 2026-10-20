import subprocess

import pytest

from mani_skill.utils.download_robot_assets import download_robot_repo

pytest.importorskip("git")


@pytest.fixture()
def local_robot_repo(tmp_path):
    """Create a local git repo with a dummy robot asset and two commits.

    Returns (path, first_commit_sha, second_commit_sha).
    """
    repo_dir = tmp_path / "src_repo"
    repo_dir.mkdir()
    (repo_dir / "panda").mkdir()
    (repo_dir / "panda" / "panda.urdf").write_text("<robot name='panda'/>")

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=repo_dir, check=True, capture_output=True
        )

    git("init", "-b", "main")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    git("add", "-A")
    git("commit", "-m", "initial robot")
    first = git("rev-parse", "HEAD").stdout.decode().strip()
    (repo_dir / "panda" / "link.urdf").write_text("<robot name='link'/>")
    git("add", "-A")
    git("commit", "-m", "add link robot")
    second = git("rev-parse", "HEAD").stdout.decode().strip()
    return repo_dir, first, second


def test_download_robot_repo_clone(tmp_path, local_robot_repo):
    src, first, second = local_robot_repo
    target = tmp_path / "robots"
    repo_path = download_robot_repo(str(src), target_dir=target)
    assert (repo_path / "panda" / "panda.urdf").exists()
    # shallow clone by default
    assert (repo_path / ".git" / "shallow").exists()


def test_download_robot_repo_pin_commit(tmp_path, local_robot_repo):
    src, first, second = local_robot_repo
    target = tmp_path / "robots"
    repo_path = download_robot_repo(str(src), commit=first, target_dir=target)
    assert (repo_path / "panda" / "panda.urdf").exists()
    assert not (repo_path / "panda" / "link.urdf").exists()

    # re-pinning to a newer commit fetches and checks it out
    download_robot_repo(str(src), commit=second, target_dir=target)
    assert (repo_path / "panda" / "link.urdf").exists()


def test_download_robot_repo_idempotent(tmp_path, local_robot_repo):
    src, first, second = local_robot_repo
    target = tmp_path / "robots"
    repo_path = download_robot_repo(str(src), commit=second, target_dir=target)
    mtime_before = (repo_path / "panda" / "panda.urdf").stat().st_mtime_ns

    # second call with the same commit is a no-op
    result = download_robot_repo(str(src), commit=second, target_dir=target)
    assert result == repo_path
    assert (repo_path / "panda" / "panda.urdf").stat().st_mtime_ns == mtime_before


def test_download_robot_repo_not_a_git_repo(tmp_path, local_robot_repo):
    src, first, second = local_robot_repo
    target = tmp_path / "robots"
    conflicting = target / "src_repo"
    conflicting.mkdir(parents=True)
    (conflicting / "README").write_text("not a repo")
    with pytest.raises(FileExistsError):
        download_robot_repo(str(src), target_dir=target)
    # force replaces it
    download_robot_repo(str(src), target_dir=target, force=True)
    assert (conflicting / "panda" / "panda.urdf").exists()
