from setuptools import find_packages, setup

__version__ = "3.0.0b15"

long_description = """ManiSkill is a powerful unified framework for robot simulation and training powered by [SAPIEN](https://sapien.ucsd.edu/), with a strong focus on manipulation skills. The entire tech stack is as open-source as possible and ManiSkill v3 is in beta release now. Among its features include:
- GPU parallelized visual data collection system. On the high end you can collect RGBD + Segmentation data at 30,000+ FPS with a 4090 GPU!
- GPU parallelized simulation, enabling high throughput state-based synthetic data collection in simulation
- GPU parallelized heteogeneous simuluation, where every parallel environment has a completely different scene/set of objects
- Example tasks cover a wide range of different robot embodiments (humanoids, mobile manipulators, single-arm robots) as well as a wide range of different tasks (table-top, drawing/cleaning, dextrous manipulation)
- Flexible and simple task building API that abstracts away much of the complex GPU memory management code via an object oriented design
- Real2sim environments for scalably evaluating real-world policies 60-100x faster via GPU simulation.

Please refer our [documentation](https://maniskill.readthedocs.io/en/latest) to learn more information."""

setup(
    name="mani_skill",
    version=__version__,
    description="ManiSkill3: A Unified Benchmark for Generalizable Manipulation Skills",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ManiSkill contributors",
    url="https://github.com/haosulab/ManiSkill",
    packages=find_packages(include=["mani_skill*"]),
    python_requires=">=3.9",
    setup_requires=["setuptools>=62.3.0"],
    install_requires=[
        "numpy>=1.22,<2.0.0",
        "scipy",
        "dacite",
        "gymnasium==0.29.1",
        "sapien==3.0.0.b1",
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
        "pytorch_kinematics==0.7.4",
        "pynvml",  # gpu monitoring
        "tyro>=0.8.5",  # nice, typed, command line arg parser
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
            "pynvml",
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
        ],
    },
)
