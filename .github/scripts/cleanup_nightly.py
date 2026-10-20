#!/usr/bin/env python3
"""Delete old nightly releases from PyPI.

PyPI storage for the ``mani_skill-nightly`` package fills up over time and a
maintainer currently has to delete old nightly builds by hand. This script
automates that: it lists all published versions of the package, keeps the most
recent N versions as well as anything uploaded within the last M days, and
deletes the rest through the PyPI project management API.

The version selection logic (:func:`select_versions_to_delete`) is kept free of
any network/IO concerns so it can be unit tested locally (see
``.github/scripts/tests/test_cleanup_nightly.py``). Run with ``--dry-run`` to
only print what would be deleted without touching PyPI.

Usage::

    python cleanup_nightly.py \
        --package mani_skill-nightly \
        --token $PYPI_TOKEN \
        --keep-most-recent 10 \
        --max-age-days 30

    # simulate only
    python cleanup_nightly.py --package mani_skill-nightly --token dummy --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone

PYPI_JSON_API = "https://pypi.org/pypi/{package}/json"
PYPI_DELETE_API = "https://pypi.org/manage/project/{package}/release/{version}/"


def parse_upload_time(upload_time: str) -> datetime:
    """Parse a PyPI ``upload_time`` string ("2026-10-20T12:34:56")."""
    return datetime.strptime(upload_time, "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=timezone.utc
    )


def select_versions_to_delete(
    releases: dict[str, str],
    now: datetime,
    keep_most_recent: int = 10,
    max_age_days: int = 30,
) -> tuple[list[str], list[str]]:
    """Decide which nightly versions to keep and which to delete.

    Args:
        releases: mapping of version -> upload time (ISO ``YYYY-MM-DDTHH:MM:SS``).
        now: the reference "current" time (injected for testability).
        keep_most_recent: always keep this many most recently uploaded versions.
        max_age_days: also keep any version uploaded within this many days.

    Returns:
        ``(keep, delete)`` lists of version strings, both sorted oldest-first.
    """
    if not releases:
        return [], []

    ordered = sorted(releases.items(), key=lambda kv: parse_upload_time(kv[1]))

    keep: set[str] = set()
    # 1. keep the N most recent uploads
    for version, _ in ordered[-keep_most_recent:]:
        keep.add(version)
    # 2. keep everything younger than max_age_days
    cutoff = now.timestamp() - max_age_days * 86400
    for version, upload_time in ordered:
        if parse_upload_time(upload_time).timestamp() > cutoff:
            keep.add(version)

    delete = [v for v, _ in ordered if v not in keep]
    return [v for v, _ in ordered if v in keep], delete


def fetch_releases(package: str) -> dict[str, str]:
    """Return {version: upload_time} for every release of ``package``."""
    url = PYPI_JSON_API.format(package=package)
    with urllib.request.urlopen(url) as resp:  # noqa: S310 (fixed https URL)
        data = json.load(resp)
    releases = {}
    for version, files in data.get("releases", {}).items():
        if not files:
            continue
        # use the newest file upload time of the release
        upload_time = max(f["upload_time"] for f in files)
        releases[version] = upload_time
    return releases


def delete_release(package: str, version: str, token: str) -> None:
    """Delete a single release via the PyPI project management API."""
    url = PYPI_DELETE_API.format(package=package, version=version)
    # the manage API expects basic auth with __token__ / API token
    import base64

    auth = base64.b64encode(f"__token__:{token}".encode()).decode()
    req = urllib.request.Request(  # noqa: S310
        url, method="DELETE", headers={"Authorization": f"Basic {auth}"}
    )
    with urllib.request.urlopen(req) as resp:  # noqa: S310
        resp.read()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="mani_skill-nightly")
    parser.add_argument("--token", required=True, help="PyPI API token")
    parser.add_argument("--keep-most-recent", type=int, default=10)
    parser.add_argument("--max-age-days", type=int, default=30)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only print which releases would be deleted",
    )
    args = parser.parse_args(argv)

    releases = fetch_releases(args.package)
    keep, delete = select_versions_to_delete(
        releases,
        now=datetime.now(timezone.utc),
        keep_most_recent=args.keep_most_recent,
        max_age_days=args.max_age_days,
    )
    print(f"{args.package}: {len(releases)} releases, keeping {len(keep)}")
    if not delete:
        print("Nothing to delete.")
        return 0

    for version in delete:
        if args.dry_run:
            print(f"[dry-run] would delete {args.package}=={version} "
                  f"(uploaded {releases[version]})")
        else:
            print(f"Deleting {args.package}=={version} (uploaded {releases[version]})")
            delete_release(args.package, version, args.token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
