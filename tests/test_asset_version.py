import json
import os

import pytest

from mani_skill import __version__
from mani_skill.utils import io_utils
from mani_skill.utils.assets.data import (
    MANIFEST_FILENAME,
    check_version_manifest,
    get_version_fingerprint,
    write_version_manifest,
)


@pytest.fixture(autouse=True)
def clear_warned_paths():
    from mani_skill.utils.assets import data

    data._skipped_version_checks.clear()
    yield
    data._skipped_version_checks.clear()


def test_get_version_fingerprint():
    fingerprint = get_version_fingerprint()
    assert fingerprint["version"] == __version__
    # commit_id is either None (non-git install) or a non-empty string
    assert fingerprint["commit_id"] is None or len(fingerprint["commit_id"]) > 0


def test_write_and_check_manifest_match(tmp_path, caplog):
    write_version_manifest(tmp_path)
    assert (tmp_path / MANIFEST_FILENAME).exists()
    stored = io_utils.load_json(tmp_path / MANIFEST_FILENAME)
    assert stored["version"] == __version__
    # Matching manifest should not warn
    check_version_manifest(tmp_path)
    assert "Version mismatch" not in caplog.text


def test_check_version_manifest_mismatch_warns_once(tmp_path, caplog):
    manifest = tmp_path / MANIFEST_FILENAME
    manifest.write_text(json.dumps({"version": "0.0.0-fake", "commit_id": None}))
    check_version_manifest(tmp_path)
    assert "Version mismatch" in caplog.text
    # A second check of the same path should be silent (warning is memoized)
    caplog.clear()
    check_version_manifest(tmp_path)
    assert "Version mismatch" not in caplog.text


def test_check_version_manifest_strict_raises(tmp_path):
    manifest = tmp_path / MANIFEST_FILENAME
    manifest.write_text(json.dumps({"version": "0.0.0-fake", "commit_id": None}))
    with pytest.raises(IOError, match="Version mismatch"):
        check_version_manifest(tmp_path, strict=True)


def test_check_version_manifest_missing_is_ignored(tmp_path, caplog):
    check_version_manifest(tmp_path)
    assert "Version mismatch" not in caplog.text


def test_check_version_manifest_skip_env_var(tmp_path, caplog):
    manifest = tmp_path / MANIFEST_FILENAME
    manifest.write_text(json.dumps({"version": "0.0.0-fake", "commit_id": None}))
    os.environ["MS_SKIP_ASSET_VERSION_CHECK"] = "1"
    try:
        check_version_manifest(tmp_path, strict=True)
    finally:
        del os.environ["MS_SKIP_ASSET_VERSION_CHECK"]
    assert "Version mismatch" not in caplog.text
