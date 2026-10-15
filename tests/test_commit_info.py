"""Verify that experiment provenance distinguishes modified and untracked files."""

import git
import pytest

import mani_skill


@pytest.mark.parametrize("show_modified_files", [False, True])
@pytest.mark.parametrize("show_untracked_files", [False, True])
def test_get_commit_info_file_lists(
    tmp_path, monkeypatch, show_modified_files, show_untracked_files
):
    repo_path = tmp_path / "experiment"
    repo = git.Repo.init(repo_path)
    tracked = repo_path / "tracked.py"
    tracked.write_text("original\n")
    repo.index.add([str(tracked)])
    author = git.Actor("Test Author", "test@example.com")
    repo.index.commit("Initial experiment", author=author, committer=author)
    tracked.write_text("modified\n")
    (repo_path / "new_experiment.py").write_text("untracked\n")
    monkeypatch.setattr(mani_skill, "PACKAGE_DIR", repo_path / "mani_skill")

    info = mani_skill.get_commit_info(
        show_modified_files=show_modified_files,
        show_untracked_files=show_untracked_files,
    )

    if show_untracked_files:
        assert info["untracked"] == ["new_experiment.py"]
    else:
        assert "untracked" not in info
    if show_modified_files:
        assert info["modified"] == ["tracked.py"]
    else:
        assert "modified" not in info
