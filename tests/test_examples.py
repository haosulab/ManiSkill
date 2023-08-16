import pathlib

import pytest

scripts = pathlib.Path(__file__, "..", "scripts").resolve().glob("*.py")


def test_demo_random_action():
    from mani_skill2.examples.demo_random_action import main, parse_args

    main(parse_args(args=["--quiet"]))


def test_demo_vec_env():
    from mani_skill2.examples.demo_vec_env import main, parse_args

    main(parse_args(args=["--quiet"]))
