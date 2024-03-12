from setuptools import find_packages, setup

__version__ = "3.0.0.dev4"

long_description = """ManiSkill2 is a unified benchmark for learning generalizable robotic manipulation skills powered by [SAPIEN](https://sapien.ucsd.edu/). **It features 20 out-of-box task families with 2000+ diverse object models and 4M+ demonstration frames**. Moreover, it empowers fast visual input learning algorithms so that **a CNN-based policy can collect samples at about 2000 FPS with 1 GPU and 16 processes on a workstation**. The benchmark can be used to study a wide range of algorithms: 2D & 3D vision-based reinforcement learning, imitation learning, sense-plan-act, etc.

Please refer our [documentation](https://haosulab.github.io/ManiSkill2) to learn more information."""

setup(
    name="mani_skill",
    version=__version__,
    description="ManiSkill3: A Unified Benchmark for Generalizable Manipulation Skills",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ManiSkill contributors",
    url="https://github.com/haosulab/ManiSkill2",
    packages=find_packages(include=["mani_skill*"]),
    python_requires=">=3.8",
    setup_requires=["setuptools>=62.3.0"],
    install_requires=[
        "numpy>=1.22",
        "scipy",
        "dacite",
        "gymnasium==0.29.1",
        "sapien==3.0.0.dev2",
        "h5py",
        "pyyaml",
        "tqdm",
        "GitPython",
        "tabulate",
        "transforms3d",
        "trimesh",
        "rtree",
        "opencv-python",
        "imageio",
        "imageio[ffmpeg]",
        "mplib>=0.1.1",
        "pytorch_kinematics",
        "huggingface_hub",  # we use HF to version control some assets/datasets more easily
    ],
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
        ],
    },
)
