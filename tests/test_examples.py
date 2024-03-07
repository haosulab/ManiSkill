import pathlib

import pytest

scripts = pathlib.Path(__file__, "..", "scripts").resolve().glob("*.py")


def test_demo_random_action():
    from mani_skill.examples.demo_random_action import main, parse_args

    main(parse_args(args=["--quiet"]))
