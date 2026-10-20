"""Unit tests for the version selection logic of cleanup_nightly.py.

Run with::

    python3 -m pytest .github/scripts/tests/test_cleanup_nightly.py
    # or, without pytest:
    python3 .github/scripts/tests/test_cleanup_nightly.py
"""

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cleanup_nightly import parse_upload_time, select_versions_to_delete

NOW = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)


def days_ago(n: int) -> str:
    """Upload time string for n days before NOW."""
    return (NOW - timedelta(days=n)).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )


class TestSelectVersionsToDelete(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(select_versions_to_delete({}, NOW), ([], []))

    def test_deletes_versions_older_than_max_age(self):
        releases = {
            "0.1.0.dev1": days_ago(90),
            "0.1.0.dev2": days_ago(40),  # older than 30 days
            "0.1.0.dev3": days_ago(10),  # within 30 days
        }
        keep, delete = select_versions_to_delete(
            releases, NOW, keep_most_recent=1, max_age_days=30
        )
        self.assertEqual(delete, ["0.1.0.dev1", "0.1.0.dev2"])
        self.assertEqual(keep, ["0.1.0.dev3"])

    def test_always_keeps_most_recent_even_if_old(self):
        releases = {f"0.1.0.dev{i}": days_ago(200 - i) for i in range(1, 6)}
        keep, delete = select_versions_to_delete(
            releases, NOW, keep_most_recent=3, max_age_days=30
        )
        self.assertEqual(delete, ["0.1.0.dev1", "0.1.0.dev2"])
        self.assertEqual(keep, ["0.1.0.dev3", "0.1.0.dev4", "0.1.0.dev5"])

    def test_age_rule_can_extend_beyond_keep_n(self):
        # dev1..dev8 all older than max_age, but 10 most recent are kept
        releases = {f"0.1.0.dev{i}": days_ago(300 - i) for i in range(1, 9)}
        keep, delete = select_versions_to_delete(
            releases, NOW, keep_most_recent=10, max_age_days=30
        )
        self.assertEqual(delete, [])
        self.assertEqual(len(keep), 8)

    def test_mixed_age_and_count(self):
        # 12 releases, one every 5 days: 30-day window covers dev5..dev12,
        # keep-most-recent=10 covers dev3..dev12 -> delete dev1, dev2
        releases = {f"0.1.0.dev{i}": days_ago(65 - (i - 1) * 5) for i in range(1, 13)}
        keep, delete = select_versions_to_delete(
            releases, NOW, keep_most_recent=10, max_age_days=30
        )
        self.assertEqual(delete, ["0.1.0.dev1", "0.1.0.dev2"])
        self.assertEqual(len(keep), 10)

    def test_boundary_version_exactly_at_cutoff_is_kept(self):
        releases = {"0.1.0.dev1": days_ago(30)}
        keep, delete = select_versions_to_delete(releases, NOW, max_age_days=30)
        # upload exactly 30 days ago is not strictly older than the cutoff
        self.assertEqual(delete, [])
        self.assertEqual(keep, ["0.1.0.dev1"])

    def test_sorted_oldest_first(self):
        releases = {
            "0.1.0.dev2": days_ago(2),
            "0.1.0.dev1": days_ago(5),
        }
        keep, delete = select_versions_to_delete(
            releases, NOW, keep_most_recent=1, max_age_days=1
        )
        self.assertEqual(keep, ["0.1.0.dev2"])
        self.assertEqual(delete, ["0.1.0.dev1"])


class TestParseUploadTime(unittest.TestCase):
    def test_roundtrip(self):
        dt = parse_upload_time("2026-10-20T12:34:56")
        self.assertEqual((dt.year, dt.month, dt.day), (2026, 10, 20))
        self.assertEqual(dt.tzinfo, timezone.utc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
