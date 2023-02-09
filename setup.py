from setuptools import find_packages, setup

long_description = """ManiSkill2 is a large-scale robotic manipulation benchmark, focusing on learning generalizable robot agents and manipulation skills. It features 2000+ diverse objects, 20 task categories, and a large-scale demonstration set in [SAPIEN](https://sapien.ucsd.edu/), a fully-physical, realistic simulator. The benchmark can be used to study 2D & 3D vision-based imitation learning, reinforcement learning, and motion planning, etc."""


def read_requirements():
    with open("requirements.txt", "r") as f:
        lines = [l.strip() for l in f.readlines()]
    install_requires = list(filter(None, lines))
    return install_requires


setup(
    name="mani_skill2",
    version="0.3.2",
    description="ManiSkill2: A Unified Benchmark for Generalizable Manipulation Skills",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ManiSkill2 contributors",
    url="https://github.com/haosulab/ManiSkill2",
    packages=find_packages(
        include=["mani_skill2*", "warp_maniskill*"],
        exclude=["warp_maniskill.warp.tests"],
    ),
    python_requires=">=3.7",
    setup_requires=["setuptools>=62.3.0"],
    install_requires=read_requirements(),
    # Glob patterns do not automatically match dotfiles
    package_data={
        "mani_skill2": ["assets/**", "envs/mpm/shader/**"],
        "warp_maniskill.warp": ["native/*", "native/nanovdb/*"],
    },
    exclude_package_data={"": ["*.convex.stl"]},
    extras_require={
        "tests": ["pytest", "black", "isort"],
        "docs": [
            "sphinx",
            "sphinx-autobuild",
            "sphinx-rtd-theme",
            # For spelling
            "sphinxcontrib.spelling",
            # Type hints support
            "sphinx-autodoc-typehints",
            # Copy button for code snippets
            "sphinx_copybutton",
            # Markdown parser
            "myst-parser",
        ],
    },
)
