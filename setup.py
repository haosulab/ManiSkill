import argparse
import datetime
import sys
from datetime import date
from pathlib import Path

from setuptools import find_packages, setup

# update this version when a new official pypi release is made
__version__ = "3.0.0b21"


def get_package_version():
    return __version__


def get_nightly_version():
    today = date.today()
    now = datetime.datetime.now()
    timing = f"{now.hour:02d}{now.minute:02d}"
    return f"{today.year}.{today.month}.{today.day}.{timing}"


def get_python_version():
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def get_dependencies():
    install_requires = [
        "numpy>=1.22",
        "scipy",
        "dacite",
        "gymnasium==0.29.1",
        "h5py",
        "pyyaml",
        "tqdm",
        "GitPython",
        "tabulate",
        "transforms3d",
        "trimesh",
        "imageio",
        "imageio[ffmpeg]",
        "mplib==0.1.1;platform_system=='Linux'",
        "fast_kinematics==0.2.2;platform_system=='Linux'",
        "IPython",
        "pytorch_kinematics==0.7.6",
        "nvidia-ml-py",  # gpu monitoring
        "tyro>=0.8.5",  # nice, typed, command line arg parser
        "huggingface_hub",  # we use HF to version control some assets/datasets more easily
        "sapien>=3.0.0;platform_system=='Linux'",
        "sapien>=3.0.0.b1;platform_system=='Windows'",
    ]
    # NOTE (stao): until sapien is uploaded to pypi with mac support, users need to install manually below as so
    # f"sapien @ https://github.com/haosulab/SAPIEN/releases/download/nightly/sapien-3.0.0.dev20250303+291f6a77-{python_version}-{python_version}-macosx_12_0_universal2.whl;platform_system=='Darwin'"
    return install_requires


def parse_args(argv):
    parser = argparse.ArgumentParser(description="ManiSkill setup.py configuration")
    parser.add_argument(
        "--package_name",
        type=str,
        default="mani_skill",
        choices=["mani_skill", "mani_skill-nightly"],
        help="the name of this output wheel. Should be either 'mani_skill' or 'mani_skill_nightly'",
    )
    return parser.parse_known_args(argv)


def main(argv):

    args, unknown = parse_args(argv)
    name = args.package_name
    is_nightly = name == "mani_skill-nightly"

    this_directory = Path(__file__).parent
    long_description = (this_directory / "README.md").read_text(encoding="utf8")

    if is_nightly:
        version = get_nightly_version()
    else:
        version = get_package_version()

    sys.argv = [sys.argv[0]] + unknown
    print(sys.argv)
    setup(
        name=name,
        version=version,
        description="ManiSkill3: A Unified Benchmark for Generalizable Manipulation Skills",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="ManiSkill contributors",
        url="https://github.com/haosulab/ManiSkill",
        packages=find_packages(include=["mani_skill*"]),
        python_requires=">=3.9",
        setup_requires=["setuptools>=62.3.0"],
        install_requires=get_dependencies(),
        # Glob patterns do not automatically match dotfiles
        package_data={
            "mani_skill": ["assets/**", "envs/**/*", "utils/**/*"],
            "warp_maniskill.warp": ["native/*", "native/nanovdb/*"],
        },
        extras_require={
            "dev": [
                "pytest",
                "black",
                "isort",
                "pre-commit",
                "build",
                "twine",
                "stable_baselines3",
                "nvidia-ml-py",
                "pytest-xdist[psutil]",
                "pytest-forked",
            ],
            "docs": [
                # Note that currently sphinx 7 does not work, so we must use v6.2.1. See https://github.com/kivy/kivy/issues/8230 which tracks this issue. Once fixed we can use a later version
                "sphinx==6.2.1",
                "sphinx-autobuild",
                "pydata_sphinx_theme",
                # For spelling
                "sphinxcontrib.spelling",
                # Type hints support
                "sphinx-autodoc-typehints",
                # Copy button for code snippets
                "sphinx_copybutton",
                # Markdown parser
                "myst-parser",
                "sphinx-subfigure",
                "sphinxcontrib-video",
                "sphinx-togglebutton",
                "sphinx_design",
                "sphinx-autoapi",
            ],
        },
    )


if __name__ == "__main__":
    main(sys.argv[1:])
