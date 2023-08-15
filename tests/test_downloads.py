import pytest


def test_download_asset_all():
    from mani_skill2.utils.download_asset import main, parse_args

    main(parse_args(args=["all", "-y", "--quiet", "-o", "/tmp/ms2test"]))


def test_download_demo_all():
    from mani_skill2.utils.download_demo import main, parse_args

    main(parse_args(args=["all", "--quiet", "-o", "/tmp/ms2test"]))
